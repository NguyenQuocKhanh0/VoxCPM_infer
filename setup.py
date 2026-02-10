import subprocess
import platform
from setuptools import setup, find_packages
from setuptools.command.install import install


def run(cmd):
    print(f"[RUN] {cmd}")
    subprocess.check_call(cmd, shell=True)


def has_cmd(cmd):
    try:
        subprocess.check_output([cmd, "--version"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
def install_espeak_ng_windows_portable():
    import os
    import urllib.request
    import subprocess

    # Release asset hiện tại của espeak-ng 1.52.0 là espeak-ng.msi (không còn zip)
    url = "https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi"

    base_dir = os.path.join(os.getcwd(), "third_party", "espeak-ng")
    os.makedirs(base_dir, exist_ok=True)

    msi_path = os.path.join(base_dir, "espeak-ng.msi")
    extract_dir = os.path.abspath(os.path.join(base_dir, "extract"))

    print("⬇️ Download espeak-ng (MSI)...")
    urllib.request.urlretrieve(url, msi_path)

    print("📦 Extract MSI (administrative install)...")
    os.makedirs(extract_dir, exist_ok=True)

    # msiexec /a = extract files to TARGETDIR (không cần cài đặt “thật”)
    cmd = f'msiexec /a "{msi_path}" /qn TARGETDIR="{extract_dir}"'
    subprocess.check_call(cmd, shell=True)

    exe_path = None
    for root, _, files in os.walk(extract_dir):
        if "espeak-ng.exe" in files:
            exe_path = os.path.join(root, "espeak-ng.exe")
            break

    if not exe_path:
        raise RuntimeError(f"❌ Không tìm thấy espeak-ng.exe sau khi extract MSI. Kiểm tra: {extract_dir}")

    print(f"✅ espeak-ng ready: {exe_path}")

    # Add vào PATH cho process hiện tại
    os.environ["PATH"] = os.path.dirname(exe_path) + os.pathsep + os.environ.get("PATH", "")

    return exe_path



import platform

def install_espeak_ng():
    if platform.system().lower() == "windows":
        install_espeak_ng_windows_portable()
    else:
        try:
            run("sudo apt install -y espeak-ng")
        except:
            run("apt install -y espeak-ng")


class CustomInstall(install):
    def run(self):
        import os
        os.environ["DS_BUILD_OPS"] = "0"
        os.environ["DS_BUILD_AIO"] = "0"

        print("=== PyTorch CUDA 12.9 ===")
        run(
            "pip install torch==2.8.0+cu129 "
            "torchvision==0.23.0+cu129 "
            "torchaudio==2.8.0+cu129 "
            "--index-url https://download.pytorch.org/whl/cu129"
        )

        print("=== torchcodec ===")
        run("pip install torchcodec==0.9")

        print("=== chunkformer (NO deepspeed) ===")
        run("pip install chunkformer --no-deps")

        print("=== deps cần cho inference ===")
        run(
            "pip install "
            "tensorboard "
            "tensorboardX "
            "textgrid "
            "langid "
            "numpy "
            "scipy "
            "soundfile "
            "voxcpm "
            "clearvoice "
            "phonemizer "
            "gradio "
            "jiwer "
            "colorama"
        )

        print("=== espeak-ng ===")
        install_espeak_ng()

        super().run()



setup(
    name="voxcpm-env",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],
    cmdclass={"install": CustomInstall},
)
