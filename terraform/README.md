# Terraform by GCCP

GCCP can generate [Terraform](https://www.terraform.io/) configuration to help
you set up a secure virtual private cloud (VPC), a secure Google Kubernetes
Engine (GKE) cluster, and additional infrastructure for generated components.

### Terraform version

The supported terraform version can be found in [versions.tf](./versions.tf).
We recommend [tfenv](https://github.com/tfutils/tfenv) to manage different
Terraform versions on your machine.

### Remote state

By default, Terraform stores state locally in a file named `terraform.tfstate`. When working with
Terraform in a team, use of a local file makes Terraform usage complicated because each user must
make sure they always have the latest state data before running Terraform and make sure that nobody
else runs Terraform at the same time. To solve this Terraform allows to store the state remotely.
In general, storing the state remotely has much more advantages than just working in a team. It 
also increases reliability in terms of backups, keeps your infrastructure secure by making the 
process of applying the Terraform updates stable and secure, and facilitates the use of Terraform 
in automated pipelines. At ML6 we believe using the remote state is the way of working with Terraform.

GCCP setup enforces the remote state by generating `backend.tf` file, which makes sure Terraform 
automatically writes the state data to GCS bucket.

Note that you need to create this bucket manually and enable versioning on it beforehand. After the 
bucket is created, no additional actions are needed.

### [Virtual Private Cloud (VPC)](https://bitbucket.org/ml6team/gccp/src/latest/examples/VPC.md/)

### [AI Platform Notebook](https://bitbucket.org/ml6team/gccp/src/latest/examples/NOTEBOOK.md/)

### [Google Kubernetes Engine (GKE)](https://bitbucket.org/ml6team/gccp/src/latest/examples/GKE.md/)


### Multiple environments

#### Terraform variables

To keep your terraform commands nice and short we recommend one file per environment.
If you want to start using multiple environments, duplicate the `project.tfvars` file and rename appropriately.

```
.                                           .
├── terraform/                              ├── terraform/
|     ├── environment/                      |     ├── environment/
|         └── project.tfvars  ----->        |         ├── sbx.tfvars
|                               └-->        |         └── dev.tfvars
|     ├── modules/                          |     ├── modules/
|     └── main.tf                           |     └── main.tf
└── ...                                     └── ...
```

You can now change variables per environment. For example you could change the project of your dev environment.

To execute terraform commands you will now provide one of the tfvars files

```bash
terraform plan --var-file environment/sbx.tfvars
```

#### Terraform workspaces

When you create resources with terraform they are recorded in a terraform state.
To have multiple environments running at the same time you will have to create multiple states,
one for each environment. To do this use the `terraform workspace` command.

### Exceptions

The terraform integration is built with ML6 building blocks in mind. However, in some specific client projects you could
encounter exceptions with regards to the standard ML6 project set-up.

#### Existing VPC

For an ML6 project, the creation of a VPC network is a necessary step when setting up the project.
In GCCP building blocks we assume that this VPC network is available and part of the Terraform state.

In case you need to work on top of an existing VPC network that was not created by GCCP (and therefore also not in the Terraform state):

1/ Delete the `modules/vpc/main.tf` and  `modules/vpc/variables.tf`  
2/ Update the `modules/vpc/outputs.tf` according the following structure and replace the exact network, subnet and IP secondary range values:

```
output "network_name" {
    value = "my-existing-network"
    description = "Network name"
}

output "subnets_names" {
    value = ["my-existing-subnet"]
    description = "Subnet name"
}

output "subnets_secondary_ranges" {
    value       = ["europe-west1-b-gke-pods", "europe-west1-b-gke-services"]
    description = "Subnet secondary ranges"
}
```

3/ Replace the 'vpc-network' module in the `main.tf` with following reference:

```
module "vpc-network" {
    source = "./modules/vpc"
}
```

To summarize, by following the steps above you actually mock a VPC module which 
just points to the relevant outputs which are required for other building blocks.


### Other modules

We maintain stand-alone ML6 supported modules at a [dedicated repository](https://bitbucket.org/ml6team/terraform-modules/src/master/).
