# Bash completion for cf_lab training/play scripts.
#
# Usage: add to your ~/.bashrc (host and/or inside the container):
#     source /workspace/cf_lab/scripts/completion.bash    # in container
#     source ~/Dropbox/uw/ayg/cf_lab/scripts/completion.bash   # on host
#
# Provides completion for:
#   - --task=<TAB>   : lists registered Isaac-* task IDs (scanned from source)
#   - --<TAB>        : lists known argparse flags for train.py / play.py
#
# The task list is cached for speed. Delete ~/.cache/cf_lab_tasks to refresh,
# or just edit a registration file (cache is invalidated by mtime).

_cf_lab_repo_root() {
    # Resolve repo root from this script's location.
    local src="${BASH_SOURCE[0]}"
    echo "$(cd "$(dirname "$src")/.." && pwd)"
}

_cf_lab_task_list() {
    local repo cache tasks_dir
    repo="$(_cf_lab_repo_root)"
    tasks_dir="$repo/source/cf_lab/cf_lab/tasks"
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/cf_lab_tasks"
    mkdir -p "$(dirname "$cache")"

    # Rebuild cache if missing or if any registration file is newer than cache.
    if [[ ! -f "$cache" ]] || [[ -n "$(find "$tasks_dir" -name '__init__.py' -newer "$cache" -print -quit 2>/dev/null)" ]]; then
        grep -rhE "^\s*id\s*=\s*[\"']Isaac-" "$tasks_dir" 2>/dev/null \
            | sed -E "s/.*[\"'](Isaac-[^\"']+)[\"'].*/\1/" \
            | sort -u > "$cache"
    fi
    cat "$cache"
}

_cf_lab_common_flags="--task --num_envs --seed --headless --video --video_length \
    --video_interval --enable_cameras --disable_fabric --device --livestream \
    --max_iterations --agent --distributed --export_io_descriptors \
    --use_pretrained_checkpoint --real-time --checkpoint --resume --logger \
    --log_project_name --experiment_name --run_name"

_cf_lab_complete() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # --task=<value>
    if [[ "$cur" == --task=* ]]; then
        local val="${cur#--task=}"
        local matches
        matches=$(compgen -W "$(_cf_lab_task_list)" -- "$val")
        COMPREPLY=()
        local m
        while IFS= read -r m; do
            [[ -n "$m" ]] && COMPREPLY+=("--task=$m")
        done <<< "$matches"
        return 0
    fi

    # --task <value>
    if [[ "$prev" == "--task" ]]; then
        COMPREPLY=( $(compgen -W "$(_cf_lab_task_list)" -- "$cur") )
        return 0
    fi

    # --<flag>
    if [[ "$cur" == --* ]]; then
        COMPREPLY=( $(compgen -W "$_cf_lab_common_flags" -- "$cur") )
        return 0
    fi

    # Fall back to filename completion (for script path, checkpoint paths, etc.).
    COMPREPLY=( $(compgen -f -- "$cur") )
}

# Bind to the script names we care about. `python` is too broad, so we register
# the wrapper via the script basename when invoked as `python <script>`.
# Bash completion can't easily key on the second argv, so we instead bind on
# `python` and dispatch based on whether one of our scripts is in the command.
_cf_lab_python_dispatch() {
    local i
    for ((i=1; i<${#COMP_WORDS[@]}; i++)); do
        case "${COMP_WORDS[i]}" in
            *scripts/rsl_rl/train.py|*scripts/rsl_rl/play.py|\
            *scripts/rl_games/train.py|*scripts/rl_games/play.py|\
            *scripts/skrl/train.py|*scripts/skrl/play.py|\
            *scripts/zero_agent.py|*scripts/random_agent.py|\
            *scripts/list_envs.py)
                _cf_lab_complete
                return 0
                ;;
        esac
    done
    # Not one of our scripts — fall back to default file completion.
    COMPREPLY=( $(compgen -f -- "${COMP_WORDS[COMP_CWORD]}") )
}

complete -o filenames -F _cf_lab_python_dispatch python python3
