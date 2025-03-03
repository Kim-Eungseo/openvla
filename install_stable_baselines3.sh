if [ -d "stable-baselines3/.git" ]; then
    echo "Repository exists, pulling latest changes..."
    cd stable-baselines3
    git pull
else
    echo "Cloning repository..."
    git clone https://github.com/Kim-Eungseo/stable-baselines3.git
    cd stable-baselines3
fi

uv pip install . --system --upgrade

cd ..
