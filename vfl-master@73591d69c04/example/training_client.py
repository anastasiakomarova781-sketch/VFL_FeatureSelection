import grpc
import argparse
import logging
from proto import active_vfl_pb2_grpc as actve_vfl, status_vfl_pb2_grpc as status_vfl
from proto.active_vfl_pb2 import *
from proto.common_vfl_pb2 import *
import time


def parse_args():
    parser = argparse.ArgumentParser(description="VFL Training Client Sample")

    parser.add_argument("--active-address", type=str, default="localhost:50000",
                        help="Address this client gRPC service listens on (host:port)")

    parser.add_argument("--active-dataset", type=str, default="active_dataset.csv",
                        help="Name of active dataset that contains target/label name")

    parser.add_argument("--passive-dataset", type=str, default="passive_dataset.csv",
                        help="Name of passive dataset for enrichment purposes")

    parser.add_argument("--match-id-name", type=str, default="id",
                        help="Name of the key column used for matching samples")

    parser.add_argument("--label-name", type=str, default="target",
                        help="Name of the target/label column for active dataset")

    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--active-features", type=str, default="",
                        help="Comma-separated list of active features names")

    parser.add_argument("--passive-features", type=str, default="",
                        help="Comma-separated list of passive features names")

    parser.add_argument("--train-params", type=str,
                        default="{\"num_trees\": 3, \"max_depth\": 3}")

    #parser.add_argument("--train-params", type=str,
    #                    default="{\"num_trees\": 1, \"max_depth\": 2, \"learning_rate\": 0.1, \"objective\": \"binary:bce\", \"num_class\": 1, \"max_bin\": 16, \"l2\": 0.1, \"l1\": 0, \"min_sample_split\": 4, \"min_leaf_node\": 2, \"min_child_weight\": 1, \"gh_pack\": false, \"split_info_pack\": false, \"hist_sub\": false, \"random_seed\": 42}")


    parser.add_argument("--model-name", type=str, default="result_model.pkl",
                        help="Name of model file name")

    return parser.parse_args()


def wait_work(stub, work_uuid, work_name="Work"):
    logging.info(f"{work_name} started. work_uuid: {work_uuid}")
    status = WorkStatus.WORK_STATUS_IN_PROGRESS
    message = ""
    while status == WorkStatus.WORK_STATUS_IN_PROGRESS:
        time.sleep(1)
        response = stub.GetStatus(
            StatusRequest(work_uuid=work_uuid))
        status = response.status
        message = response.message
        logging.info(
            f"{work_name} status: [{WorkStatus.Name(status)} : {message}]")

    logging.info(
        f"{work_name} finished: id: [{work_uuid}] status: [{WorkStatus.Name(status)} : {message}]")


def main(args):
    channel = grpc.insecure_channel(args.active_address)
    loading_stub = actve_vfl.DatasetLoadingStub(channel)
    training_stub = actve_vfl.TrainingStub(channel)
    status_stub = status_vfl.StatusServiceStub(channel)
    active_features = [f.strip()
                       for f in args.active_features.split(",")] if args.active_features else []
    passive_features = [f.strip()
                       for f in args.active_features.split(",")] if args.passive_features else []

    # rpc UploadActiveDataset(UploadActiveDatasetRequest) returns (WorkResponse);
    response = loading_stub.UploadActiveDataset(
        UploadDatasetRequest(
            dataset=DatasetInfo(
                dataset_name=args.active_dataset,
                match_id_name=args.match_id_name,
                dtype=args.dtype,
                label_name=args.label_name,
                features=active_features
            )
        )
    )
    work_uuid = response.work_uuid
    wait_work(status_stub, work_uuid, "Upload active dataset")

    # rpc UploadPassiveDataset(UploadPassiveDatasetRequest) returns(WorkResponse)
    response = loading_stub.UploadPassiveDataset(
        UploadDatasetRequest(
            dataset=DatasetInfo(
                dataset_name=args.passive_dataset,
                match_id_name=args.match_id_name,
                dtype=args.dtype,
                features=passive_features
            )
        )
    )
    work_uuid = response.work_uuid
    wait_work(status_stub, work_uuid, "Upload passive dataset")

    # rpc StartTraining(StartTrainingRequest) returns (WorkResponse);
    train_params = TrainingParams(params=args.train_params)

    active_request = TrainingRequest(
        params=train_params,
        dataset_name=args.active_dataset,
        output_model_name=args.model_name
    )

    passive_request = TrainingRequest(
        params=train_params,
        dataset_name=args.passive_dataset,
        output_model_name=args.model_name
    )

    request = StartTrainingRequest(
        active_request=active_request,
        passive_request=passive_request
    )

    response = training_stub.StartTraining(request)
    work_uuid = response.work_uuid
    wait_work(status_stub, work_uuid, "Training")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()
    main(args)
