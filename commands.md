# build docker images
docker build -f Dockerfile.rag-func-call -t rag-func-call .
docker build -f Dockerfile.proj -t proj .

# remove proj image
docker rmi proj

# byobu
byobu
byobu list-sessions
byobu attach-session -t 1