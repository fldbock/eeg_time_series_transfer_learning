
/******************************************
	Remote backend configuration
 *****************************************/

# setup of the backend gcs bucket that will keep the remote state

# Note you need to create this bucket manually beforehand
# run the following commands to create a bucket + enable versioning
# gsutil mb -l europe-west1 -p arctic-window-318309 gs://arctic-window-318309_terraform
# gsutil versioning set on gs://arctic-window-318309_terraform

terraform {
    backend "gcs" {
        bucket  = "arctic-window-318309_terraform"
        prefix  = "terraform/state"
    }
}
