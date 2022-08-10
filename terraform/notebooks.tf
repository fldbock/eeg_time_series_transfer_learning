
data "google_compute_network" "network" {
  name = module.vpc-network.network_name
}

data "google_compute_subnetwork" "subnetwork" {
  name   = module.vpc-network.subnets_names[0]
}

resource "google_project_service" "notebook" {
  project = var.project
  service   = "notebooks.googleapis.com"
  disable_dependent_services = true
}

resource "google_notebooks_instance" "instance" {
  name = "flor-vm-cpu"
  location = var.zone
  machine_type = "e2-medium"


  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf2-2-2-cu101-notebooks"
  }

  no_public_ip = true

  network = data.google_compute_network.network.id
  subnet = data.google_compute_subnetwork.subnetwork.id

  depends_on = [google_project_service.notebook]
}

resource "google_notebooks_instance" "instance_gpu" {
  name         = "flor-vm-gpu"
  location     = var.zone
  machine_type = "n1-standard-4"

  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf2-2-2-cu101-notebooks"
  }

  no_public_ip = true

  network = data.google_compute_network.network.id
  subnet  = data.google_compute_subnetwork.subnetwork.id

  depends_on = [google_project_service.notebook]

  install_gpu_driver = true
  accelerator_config {
    type       = "NVIDIA_TESLA_K80"
    core_count = 1
  }
}

resource "google_notebooks_instance" "instance_gpu_large" {
  name         = "flor-vm-gpu-large"
  location     = var.zone
  machine_type = "n1-standard-4"

  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf2-2-2-cu101-notebooks"
  }

  no_public_ip = true

  network = data.google_compute_network.network.id
  subnet  = data.google_compute_subnetwork.subnetwork.id

  depends_on = [google_project_service.notebook]

  boot_disk_type = "PD_BALANCED"
  boot_disk_size_gb = 300

  install_gpu_driver = true
  accelerator_config {
    type       = "NVIDIA_TESLA_K80"
    core_count = 1
  }
}

resource "google_notebooks_instance" "instance_gpu_large2" {
  name         = "flor-vm-gpu-large2"
  location     = var.zone
  machine_type = "n1-standard-4"

  vm_image {
    project      = "deeplearning-platform-release"
    image_family = "tf2-2-2-cu101-notebooks"
  }

  no_public_ip = true

  network = data.google_compute_network.network.id
  subnet  = data.google_compute_subnetwork.subnetwork.id

  depends_on = [google_project_service.notebook]

  data_disk_size_gb = 500

  install_gpu_driver = true
  accelerator_config {
    type       = "NVIDIA_TESLA_K80"
    core_count = 1
  }
}
