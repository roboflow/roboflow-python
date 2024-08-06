import json

from roboflow.adapters import deploymentapi
from roboflow.config import load_roboflow_api_key


def add_deployment_parser(subparsers):
    deployment_parser = subparsers.add_parser(
        "deployment",
        help="deployment related commands.  type 'roboflow deployment' to see detailed command help",
    )
    deployment_subparsers = deployment_parser.add_subparsers(title="deployment subcommands")
    deployment_add_parser = deployment_subparsers.add_parser("add", help="create a dedicated deployment")
    deployment_get_parser = deployment_subparsers.add_parser(
        "get", help="show detailed info for a dedicated deployment"
    )
    deployment_list_parser = deployment_subparsers.add_parser("list", help="list dedicated deployments in a workspace")
    deployment_delete_parser = deployment_subparsers.add_parser("delete", help="delete a dedicated deployment")

    deployment_add_parser.set_defaults(func=add_deployment)
    deployment_add_parser.add_argument("-a", dest="api_key", help="api key")
    deployment_add_parser.add_argument(
        "-s", dest="security_level", help="security level (protected)", default="protected"
    )
    deployment_add_parser.add_argument(
        "-m",
        dest="machine_type",
        help="machine type (gcp-n2-cpu, gcp-t4-gpu, gcp-l4-gpu)",
        default="gcp-n2-cpu",
    )
    deployment_add_parser.add_argument("-n", dest="deployment_name", help="deployment name")
    deployment_add_parser.add_argument(
        "-v", dest="inference_version", help="inference server version (default: latest)", default="latest"
    )

    deployment_get_parser.set_defaults(func=get_deployment)
    deployment_get_parser.add_argument("-a", dest="api_key", help="api key")
    deployment_get_parser.add_argument("-d", dest="deployment_id", help="deployment id")

    deployment_list_parser.set_defaults(func=list_deployment)
    deployment_list_parser.add_argument("-a", dest="api_key", help="api key")

    deployment_delete_parser.set_defaults(func=delete_deployment)
    deployment_delete_parser.add_argument("-a", dest="api_key", help="api key")
    deployment_delete_parser.add_argument("-d", dest="deployment_id", help="deployment id")


def add_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    ret_json = deploymentapi.add_deployment(
        api_key,
        args.security_level,
        args.machine_type,
        args.deployment_name,
        args.inference_version,
    )
    print(json.dumps(ret_json, indent=2))


def get_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    ret_json = deploymentapi.get_deployment(api_key, args.deployment_id)
    print(json.dumps(ret_json, indent=2))


def list_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    ret_json = deploymentapi.list_deployment(api_key)
    print(json.dumps(ret_json, indent=2))


def delete_deployment(args):
    api_key = args.api_key or load_roboflow_api_key(None)
    ret_json = deploymentapi.delete_deployment(api_key, args.deployment_id)
    print(json.dumps(ret_json, indent=2))
