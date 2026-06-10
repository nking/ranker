import os
import re
import subprocess
from pathlib import Path
import urllib.request
import platform
import stat
import tarfile
from typing import List


def image_exists_on_host(docker_path:str, image_name: str) -> bool:
    """Checks if the image exists in the host machine's Docker cache."""
    res = subprocess.run([docker_path, "image", "inspect", image_name],
        capture_output=True)
    return res.returncode == 0

def image_exists_on_node(docker_path:str, node_name: str, image_name: str) -> bool:
    """Checks if the image is already loaded inside the node's containerd storage."""
    # Lists image names inside the k8s.io namespace of the specific node
    res = subprocess.run(
        [docker_path, "exec", node_name, "ctr", "-n", "k8s.io", "images", "ls",
            "-q"], capture_output=True, text=True
    )
    return image_name in res.stdout

def sideload_image_without_kind(docker_path:str, kind_path:str, image_name: str, cluster_name: str):
    """
    Smart-caches and pipes images directly into containerd on every node.
    Safe for local-only images and 100% network-immune if cached.
    """
    print(f"\n🐳 Processing: {image_name}")
    
    #  Handle Host-Level Caching / Pulling
    if image_exists_on_host(docker_path=docker_path, image_name=image_name):
        print("  ✅ Image found locally on host. Skipping remote network pull.")
    else:
        # If it's a remote image missing locally, fetch it
        print("  📥 Image missing on host. Pulling from registry...")
        try:
            subprocess.run([docker_path, "pull", image_name], check=True,
                stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print( f"  ❌ Error: Could not pull '{image_name}' and it's not found locally.")
            print( "     If this is a local build image, build it on the host first.")
            return
    
    #  Discover cluster nodes
    try:
        nodes_output = subprocess.run(
            [kind_path, "get", "nodes", "--name", cluster_name],
            capture_output=True, text=True, check=True
        )
        nodes = [n.strip() for n in nodes_output.stdout.strip().split('\n') if
            n.strip()]
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️ Could not query kind nodes: {e}")
        return
    
    # Stream to nodes if missing internally
    for node in nodes:
        if image_exists_on_node(docker_path=docker_path, node_name=node, image_name=image_name):
            print(f"  ✅ Already present inside containerd on node: {node}")
            continue
        
        print(f"  📦 Piping directly into containerd on node: {node}...")
        
        save_process = subprocess.Popen([docker_path, "save", image_name],
            stdout=subprocess.PIPE)
        exec_process = subprocess.Popen(
            [docker_path, "exec", "-i", node, "ctr", "-n", "k8s.io", "images",
                "import", "-"],
            stdin=save_process.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        save_process.stdout.close()
        _, stderr_data = exec_process.communicate()
        
        if exec_process.returncode != 0:
            print(f"  ⚠️ Failed on {node}. Details: {stderr_data.decode().strip()}")
        else:
            print(f"  ✅ Successfully streamed to {node}")

def delete_namespace_kubeflow(kubectl_path):
    """Deletes the kubeflow namespace to prevent dirty storage crashes."""
    print("🧹 Deleting old kubeflow namespace if it exists...")
    subprocess.run(
        [kubectl_path, "delete", "namespace", "kubeflow",
            "--ignore-not-found=true"],
        check=True
    )
    print("✅ Slate wiped clean.")


def vendor_manifests(git_path:str, vendor_dir: Path, kfp_version: str):
    """Clones the specific pinned version of KFP manifests locally if they don't exist."""
    vendor_dir.mkdir(parents=True, exist_ok=True)
    
    repos = {
        "kubeflow-pipelines": "https://github.com/kubeflow/pipelines.git"
    }
    
    for name, url in repos.items():
        repo_path = vendor_dir / name
        if not repo_path.exists():
            print(f"📥 Vendoring {name} (Version: {kfp_version}) locally...")
            # Enforce the explicit version checkout tag using the --branch argument
            subprocess.run([
                git_path, "clone", "--depth", "1", "--branch", kfp_version, url, str(repo_path)
            ], check=True)
        else:
            print(f"✅ {name} already vendored at {repo_path}.")


def extract_and_sideload_images(kustomize_path:str, docker_path:str, kind_path:str, vendor_dirs: List[Path], cluster_name: str):
    """
        Builds the target Kustomize manifests in memory, extracts the exact container
        images required for deployment, and sideloads them offline.
        """
    print("🔍 Compiling target manifests to determine required container images...")
    
    image_pattern = re.compile(
        r'^\s*image:\s*["\']?([^"\']+\/[^"\']+:[^"\']+|[^"\':\s]+:[^"\':\s]+)["\']?',
        re.MULTILINE)
    found_images = set()
    
    for vendor_dir in vendor_dirs:
        try:
            # 1. Compile the manifests to get the final, actual YAML
            build = subprocess.run(
                [kustomize_path, "build", str(vendor_dir)],
                capture_output=True, text=True, check=True
            )
            
            # 2. Scan the compiled YAML string line by line
            for line in build.stdout.splitlines():
                match = image_pattern.search(line)
                if match:
                    image = match.group(1).strip()
                    # Failsafe: Ignore anything with leftover template variables
                    if not any(char in image for char in ['{', '$', '(', '<']):
                        found_images.add(image)
        
        except subprocess.CalledProcessError as e:
            print(
                f"❌ Failed to build kustomize directory {vendor_dir}: {e.stderr}")
            raise
    
    print(
        f"🎯 Found {len(found_images)} unique container images required for this exact deployment.")
    
    # Process each discovered image through the smart-cache pipeline
    for img in sorted(found_images):
        sideload_image_without_kind(docker_path=docker_path, kind_path=kind_path,  image_name=img, cluster_name=cluster_name)
    
    print("\n🏁 All required KFP images successfully pre-loaded!")

def deploy_offline(kustomize_path:str, kubectl_path:str, deploy_dir: Path):
    """Builds and applies manifests completely offline using kustomize then kubectl."""
    print(f"🚀 Deploying KFP components from: {deploy_dir}")
    
    build = subprocess.run(
        [kustomize_path, "build", str(deploy_dir)],
        capture_output=True, text=True, check=True
    )
    
    #the wait flag makes this synchronous
    subprocess.run(
        [kubectl_path, "apply", "-f", "-", "--wait"],
        input=build.stdout, text=True, check=True
    )

def build_kustomize_if_not_found(kustomize_path: str, version: str = "v5.8.1"):
    """
    Natively detects system OS and architecture, downloads the exact Kustomize
    tarball directly from GitHub Releases, and extracts it without using bash scripts.
    """
    if os.path.exists(kustomize_path):
        return
    
    dir_path = os.path.dirname(kustomize_path)
    
    # Detect Operating System natively
    sys_os = platform.system().lower()
    if sys_os not in ["linux", "darwin"]:
        raise RuntimeError(
            f"❌ Unsupported operating system for this automated setup: {sys_os}")
    
    # Detect CPU Architecture natively
    machine = platform.machine().lower()
    if machine in ["x86_64", "amd64"]:
        arch = "amd64"
    elif machine in ["arm64", "aarch64"]:
        arch = "arm64"
    else:
        raise RuntimeError(f"❌ Unsupported machine architecture: {machine}")
    
    # Construct the exact explicit asset URL
    # Kustomize release tags use a URL-encoded slash: kustomize%2FvX.X.X
    archive_name = f"kustomize_{version}_{sys_os}_{arch}.tar.gz"
    url = f"https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2F{version}/{archive_name}"
    
    archive_path = os.path.join(dir_path, archive_name)
    
    print(f"📥 Downloading Kustomize {version} directly for {sys_os}-{arch}...")
    try:
        # Download the compressed tarball natively
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=90) as response, open(archive_path, 'wb') as out_file:
            out_file.write(response.read())
        
        # Extract only the 'kustomize' binary from inside the archive
        print("📦 Extracting binary...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extract("kustomize", path=dir_path)
        
        # If the expected kustomize_path name differs from 'kustomize', rename it
        extracted_binary = os.path.join(dir_path, "kustomize")
        if extracted_binary != kustomize_path:
            os.rename(extracted_binary, kustomize_path)
        
        # Make the binary executable (equivalent to chmod +x)
        st = os.stat(kustomize_path)
        os.chmod(kustomize_path, st.st_mode | stat.S_IEXEC)
        print(f"🚀 Kustomize successfully installed to {kustomize_path}")
    
    except Exception as e:
        print(f"❌ Failed to download or extract Kustomize: {e}")
        raise
    finally:
        # Clean up the downloaded temporary tarball
        if os.path.exists(archive_path):
            os.remove(archive_path)