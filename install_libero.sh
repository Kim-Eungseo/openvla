if [ -d "LIBERO/.git" ]; then
    echo "LIBERO repository exists, pulling latest changes..."
    cd LIBERO
    git pull
else
    echo "Cloning LIBERO repository..."
    git clone https://github.com/Kim-Eungseo/LIBERO.git
    cd LIBERO
fi

uv pip install . --system --upgrade

cd ../experiments/robot/libero
uv pip install -r libero_requirements.txt --system --upgrade

cd ../../../
