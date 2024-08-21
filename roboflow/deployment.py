import json
import time
from datetime import datetime

from roboflow.adapters import deploymentapi
from roboflow.config import load_roboflow_api_key


def add_deployment_parser(subparsers):
    deployment_parser = subparsers.add_parser(
        "deployment",
        help="deployment related commands.  type 'roboflow deployment' to see detailed command help",
    )
    deployment_subparsers = deployment_parser.add_subparsers(title="deployment subcommands")
    deployment_machine_type_parser = deployment_subparsers.add_parser("machine_type", help="list machine types")
    deployment_add_parser = deployment_subparsers.add_parser("add", help="create a dedicated deployment")
    deployment_get_parser = deployment_subparsers.add_parser(
        "get", help="show detailed info for a dedicated deployment"
    )
    deployment_list_parser = deployment_subparsers.add_parser("list", help="list dedicated deployments in a workspace")
    deployment_delete_parser = deployment_subparsers.add_parser("delete", help="delete a dedicated deployment")

    deployment_machine_type_parser.set_defaults(func=list_machine_types)
    deployment_machine_type_parser.add_argument("-a", "--api_key", help="api key")

    deployment_add_parser.set_defaults(func=add_deployment)
    deployment_add_parser.add_argument("-a", "--api_key", help="api key")
    deployment_add_parser.add_argument(
        "deployment_name",
        help="deployment name, must contain 5-15 lowercase characters, first character must be a letter",
    )
    # deployment_add_parser.add_argument(
    #     "-s", "--security_level", help="security level (protected)", default="protected"
    # )
    deployment_add_parser.add_argument(
        "-m", "--machine_type", help="machine type, run `roboflow deployment machine_type` to see available options"
    )
    deployment_add_parser.add_argument(
        "-t",
        "--duration",
        help="duration, how long you want to keep the deployment (unit: hour, default: 3)",
        type=float,
        default=3,
    )
    deployment_add_parser.add_argument(
        "-e", "--no_delete_on_expiration", help="keep when expired (default: False)", action="store_true"
    )
    deployment_add_parser.add_argument(
        "-v",
        "--inference_version",
        help="inference server version (default: latest)",
        default="latest",
    )
    deployment_add_parser.add_argument(
        "-w", "--wait_on_pending", help="wait if deployment is pending", action="store_true"
    )

    deployment_get_parser.set_defaults(func=get_deployment)
    deployment_get_parser.add_argument("-a", "--api_key", help="api key")
    deployment_get_parser.add_argument("deployment_name", help="deployment name")
    deployment_get_parser.add_argument(
        "-w", "--wait_on_pending", help="wait if deployment is pending", action="store_true"
    )

    deployment_list_parser.set_defaults(func=list_deployment)
    deployment_list_parser.add_argument("-a", "--api_key", help="api key")

    deployment_delete_parser.set_defaults(func=delete_deployment)
    deployment_delete_parser.add_argument("-a", "--api_key", help="api key")
    deployment_delete_parser.add_argument("deployment_name", help="deployment name")


def list_machine_types(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    if api_key is None:
        print("Please provide an api key")
        return
    status_code, msg = deploymentapi.list_machine_types(api_key)
    if status_code != 200:
        print(f"{status_code}: {msg}")
        return
    print(json.dumps(msg, indent=2))


def add_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    if api_key is None:
        print("Please provide an api key")
        return
    status_code, msg = deploymentapi.add_deployment(
        api_key,
        # args.security_level,
        args.machine_type,
        args.duration,
        (not args.no_delete_on_expiration),
        args.deployment_name,
        args.inference_version,
    )

    if status_code != 200:
        print(f"{status_code}: {msg}")
        return
    else:
        print(f"Deployment {args.deployment_name} created successfully")
        print(json.dumps(msg, indent=2))

    if args.wait_on_pending:
        get_deployment(args)


def get_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    if api_key is None:
        print("Please provide an api key")
        return
    while True:
        status_code, msg = deploymentapi.get_deployment(api_key, args.deployment_name)
        if status_code != 200:
            print(f"{status_code}: {msg}")
            return

        if (not args.wait_on_pending) or msg["status"] != "pending":
            print(json.dumps(msg, indent=2))
            break

        print(f'{datetime.now().strftime("%H:%M:%S")} Waiting for deployment {args.deployment_name} to be ready...\n')
        time.sleep(30)


def list_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    if api_key is None:
        print("Please provide an api key")
        return
    status_code, msg = deploymentapi.list_deployment(api_key)
    if status_code != 200:
        print(f"{status_code}: {msg}")
        return
    print(json.dumps(msg, indent=2))


def delete_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    if api_key is None:
        print("Please provide an api key")
        return
    status_code, msg = deploymentapi.delete_deployment(api_key, args.deployment_name)
    if status_code != 200:
        print(f"{status_code}: {msg}")
        return
    print(json.dumps(msg, indent=2))
