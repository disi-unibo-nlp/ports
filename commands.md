# build docker images
docker build -f Dockerfile.rag-func-call -t rag-func-call .
docker build -f Dockerfile.proj -t proj .

# remove proj image
docker rmi proj