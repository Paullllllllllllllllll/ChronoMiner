"""Wait for the leases of a run that no longer renews them.

A live run renews every lease it holds each ``HEARTBEAT_INTERVAL`` seconds,
which moves the lease's expiry forward. A killed run leaves its leases as
they were until they expire, at most ``LEASE_TTL`` seconds after its last
renewal. :func:`wait_for_stale_leases` takes the leases another run holds on
a run's items and reads them again every :data:`RENEWAL_CHECK` seconds, or
just after the latest expiry when that comes first. A lease whose expiry
moved or whose holder changed belongs to a live run and is no longer waited
for; a lease that is gone has been released or has expired. The wait ends
when every lease is either gone or found live, so it lasts at most
``LEASE_TTL`` seconds plus one check. The clock and the sleep are
injectable, so tests run the loop without waiting.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from .store import HEARTBEAT_INTERVAL

__all__ = ["RENEWAL_CHECK", "HeldLease", "wait_for_stale_leases"]

logger = logging.getLogger(__name__)

RENEWAL_CHECK: Final = 1.5 * HEARTBEAT_INTERVAL
"""Seconds between two reads; a live holder renews at least once between them."""

_MARGIN: Final = 1.0
"""Seconds past the latest expiry at which the leases are read again."""


@dataclass(frozen=True)
class HeldLease:
    """A lease another run holds on one item: the item's key, holder and expiry.

    *until* is the expiry in seconds since the epoch.
    """

    key: str
    holder: str | None
    until: float


def _stamp(until: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(until))


def wait_for_stale_leases(
    held: Sequence[HeldLease],
    read: Callable[[], Sequence[HeldLease]],
    *,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until no lease of *held* is left that its holder stopped renewing.

    *read* returns the leases other runs hold now, for example from a new
    plan; it is called only when *held* is not empty. ``KeyboardInterrupt``
    propagates.
    """
    watched = {lease.key: lease for lease in held}
    if not watched:
        return
    latest = max(lease.until for lease in watched.values())
    logger.warning(
        "%d item(s) leased by another run. Waiting until %s at the latest for "
        "leases their run no longer renews; an item whose lease is renewed is "
        "skipped.",
        len(watched),
        _stamp(latest + _MARGIN),
    )
    while watched:
        latest = max(lease.until for lease in watched.values())
        sleep(max(_MARGIN, min(RENEWAL_CHECK, latest + _MARGIN - clock())))
        current = {lease.key: lease for lease in read()}
        remaining: dict[str, HeldLease] = {}
        for key, lease in watched.items():
            now = current.get(key)
            if now is None:
                continue
            if now.holder != lease.holder or now.until > lease.until:
                logger.info("Run %s still renews its lease on %s.", now.holder, key)
                continue
            remaining[key] = lease
        watched = remaining
