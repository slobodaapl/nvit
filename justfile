#!/usr/bin/env just --justfile

# tmux session name
session := env_var_or_default("SESSION", "nvit")

default:
    @just --list

install:
    poetry install

train-local *ARGS:
    #!/usr/bin/env bash
    use_tmux=false
    args=""

    for arg in {{ARGS}}; do
        if [ "$arg" = "-d" ] || [ "$arg" = "--detach" ]; then
            use_tmux=true
        else
            args="$args $arg"
        fi
    done

    if [ "$use_tmux" = true ]; then
        if ! tmux has-session -t {{session}} 2>/dev/null; then
            tmux new-session -d -s {{session}}
        fi
        tmux send-keys -t {{session}} "./launcher.sh $args" Enter
        echo "Training started in tmux session '{{session}}'. Use 'tmux attach -t {{session}}' to view."
        echo "When attached, press Ctrl+B, D to detach from the session."
    else
        ./launcher.sh $args
    fi

docker-build:
    cd docker && ./build.sh

train *ARGS:
    #!/usr/bin/env bash
    use_tmux=false
    args=""

    for arg in {{ARGS}}; do
        if [ "$arg" = "-d" ] || [ "$arg" = "--detach" ]; then
            use_tmux=true
        else
            args="$args $arg"
        fi
    done
    
    cmd="./docker_launcher.sh $args"
    
    if [ "$use_tmux" = true ]; then
        if ! tmux has-session -t {{session}} 2>/dev/null; then
            tmux new-session -d -s {{session}}
        fi
        tmux send-keys -t {{session}} "$cmd" Enter
        echo "Training started in tmux session '{{session}}'. Use 'tmux attach -t {{session}}' to view."
        echo "When attached, press Ctrl+B, D to detach from the session."
    else
        $cmd
    fi

run-profiles *ARGS:
    #!/usr/bin/env bash
    use_tmux=false
    args=""

    for arg in {{ARGS}}; do
        if [ "$arg" = "-d" ] || [ "$arg" = "--detach" ]; then
            use_tmux=true
        else
            args="$args $arg"
        fi
    done

    if [ "$use_tmux" = true ]; then
        if ! tmux has-session -t {{session}} 2>/dev/null; then
            tmux new-session -d -s {{session}}
        fi
        tmux send-keys -t {{session}} "./run_profiles.sh $args" Enter
        echo "Profile training started in tmux session '{{session}}'. Use 'tmux attach -t {{session}}' to view."
        echo "When attached, press Ctrl+B, D to detach from the session."
    else
        ./run_profiles.sh $args
    fi

clean:
    rm -rf .cache
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

attach:
    tmux attach -t {{session}}
