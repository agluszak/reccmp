import reccmp.color


def format_address(addr: int) -> str:
    """This is here just to document each spot where we
    convert an int address into a string.
    In the future, the format may be customizable. (GH #370)"""
    return f"{addr:#x}"


def print_diff(udiff):
    """Print diff in difflib.unified_diff format."""
    if udiff is None:
        return False

    has_diff = False
    for line in udiff:
        has_diff = True
        color = ""
        if line.startswith("++") or line.startswith("@@") or line.startswith("--"):
            # Skip unneeded parts of the diff for the brief view
            continue
        # Work out color if we are printing color
        if line.startswith("+"):
            color = reccmp.color.Fore.GREEN
        elif line.startswith("-"):
            color = reccmp.color.Fore.RED
        print(f"{color}{line}{reccmp.color.Style.RESET_ALL}")
    return has_diff
