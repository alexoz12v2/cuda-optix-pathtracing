# .cmake-format.py
# Compatible with cmake-format <=0.6.13 (Python config style)
# Compact style: inline FILE_SET TYPE HEADERS, break only BASE_DIRS and FILES

with section("format"):

    # Disable formatting entirely? False = enable formatting
    disable = False

    # Maximum line width
    line_width = 120

    # Indentation settings
    tab_size = 2
    use_tabchars = False

    # Separate control/command names from parentheses
    separate_ctrl_name_with_space = True
    separate_fn_name_with_space = False

    # Dangle closing parentheses
    dangle_parens = False
    dangle_align = 'prefix'

    # Horizontal wrapping limits
    max_subgroups_hwrap = 2   # number of subgroups before forcing vertical layout
    max_pargs_hwrap = 6       # number of positional args before forcing vertical layout
    max_rows_cmdline = 2      # lines allowed without nesting before force wrap

    # Align and spacing
    min_prefix_chars = 4
    max_prefix_chars = 10

    # Force wrapping only for BASE_DIRS and FILES
    always_wrap = ['BASE_DIRS', 'FILES']

    # Line endings
    line_ending = 'unix'

    # Command / keyword casing
    command_case = 'canonical'
    keyword_case = 'unchanged'

    # Enable sorting for lists marked sortable
    enable_sort = True
    autosort = False

# Optionally, keep parse and lint sections as default
with section("parse"):
    additional_commands = {}
    override_spec = {}
    vartags = []
    proptags = []

with section("lint"):
    disabled_codes = ['C0113']
