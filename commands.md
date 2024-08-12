# build docker images
docker build -f Dockerfile.rag-func-call -t rag-func-call .

# remove proj image
docker rmi proj

# byobu
byobu
byobu list-sessions
byobu attach-session -t 1

# sweeps
wandb sweep sweeps_file.yaml
wandb agent sweep_id
wandb sweep --stop entity/project/sweep_ID

# get gpu usages
scontrol show job -d | grep "IDX\|UserId"

# get user that started process
ps -o user= -p PID