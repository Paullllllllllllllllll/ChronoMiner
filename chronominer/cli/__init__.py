"""Command-line front end of ChronoMiner.

``cli.parser`` builds the top-level parser and one parser per subcommand
from the option tables; ``cli.extract``, ``cli.ranges``, ``cli.batch`` and
``cli.store`` run the subcommands and ``cli.command`` holds their shared
plumbing.
"""
