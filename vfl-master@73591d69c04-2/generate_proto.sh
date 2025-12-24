mkdir -p ./generated
python3 -m grpc_tools.protoc -Iproto=./contracts/vfl/proto --python_out=./generated --grpc_python_out=./generated ./contracts/vfl/proto/common_vfl.proto
python3 -m grpc_tools.protoc -Iproto=./contracts/vfl/proto --python_out=./generated --grpc_python_out=./generated ./contracts/vfl/proto/status_vfl.proto
python3 -m grpc_tools.protoc -Iproto=./contracts/vfl/proto --python_out=./generated --grpc_python_out=./generated ./contracts/vfl/proto/active_vfl.proto
python3 -m grpc_tools.protoc -Iproto=./contracts/vfl/proto --python_out=./generated --grpc_python_out=./generated ./contracts/vfl/proto/passive_vfl.proto

rm -rf ./example/proto
rm -rf ./python/proto
mkdir -p ./example/proto
mkdir -p ./python/proto

cp -r ./generated/proto/* ./example/proto/
cp -r ./generated/proto/* ./python/proto/
