This repository contains a minimal FastAPI backend implementing the FGSM adversarial attack using PyTorch and a Next.js frontend to demo the attack on uploaded images.

### Prerequisites
- Python 3.10+ (recommended)
- Node.js 18+

### Backend: Run Locally
1. Create and activate a virtual environment.
2. Install dependencies:
```
pip install -r backend/requirements.txt
```
3. Start the API:
```
python -m uvicorn backend.app_fgsm:app --reload --host 0.0.0.0 --port 8000
```
4. Test with curl:
```
curl -X POST "http://localhost:8000/attack" \
  -F "epsilon=0.1" \
  -F "image=@path/to/image.jpg"
```

### Frontend: Run Locally
Steps will appear after the frontend is added. You will set NEXT_PUBLIC_API_URL to point at your backend.

### FGSM (1–2 paragraphs)
Fast Gradient Sign Method perturbs an input in the direction that maximally increases the model's loss, using the sign of the gradient with respect to the input. Given an input x and label y, and loss L(θ, x, y), FGSM forms x_adv = x + ε * sign(∇_x L(θ, x, y)). With small ε, the image looks visually unchanged to humans, yet can cause misclassification. Increasing ε typically increases attack success but can introduce visible artifacts.

### Observations (fill in after evaluation)
- Note the clean vs adversarial predictions and how they change as ε grows.
- Report the accuracy drop or success rate from `backend/results_fgsm.csv`.

### Deployment (AWS)
- Backend (recommended): EC2 t2.micro with Uvicorn + Nginx
- Frontend: Amplify Hosting

#### Backend on EC2 (t2.micro)
1) Launch EC2 (Amazon Linux 2023), open inbound 80 (HTTP) and 22 (SSH). Optionally 8000 for direct testing.
2) SSH into the instance:
```
ssh -i your-key.pem ec2-user@EC2_PUBLIC_DNS
```
3) Install dependencies:
```
sudo dnf update -y
sudo dnf install -y python3.11 python3.11-pip git nginx
python3.11 -m venv venv
source venv/bin/activate
git clone <your_repo_url> app && cd app
pip install -r backend/requirements.txt
```
4) Test run:
```
python -m uvicorn backend.app_fgsm:app --host 0.0.0.0 --port 8000
```
5) Nginx reverse proxy to Uvicorn (serve on port 80):
```
sudo tee /etc/nginx/conf.d/fgsm.conf >/dev/null <<'NGINX'
server {
    listen 80;
    server_name _;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX
sudo nginx -t && sudo systemctl restart nginx
```
6) Run as a service (optional):
```
sudo tee /etc/systemd/system/fgsm.service >/dev/null <<'UNIT'
[Unit]
Description=FGSM FastAPI Service
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/app
Environment="PATH=/home/ec2-user/venv/bin"
ExecStart=/home/ec2-user/venv/bin/uvicorn backend.app_fgsm:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now fgsm
```
7) Note the public backend URL: `http://EC2_PUBLIC_DNS/`.

#### Frontend on Amplify Hosting
1) Push this repo to GitHub.
2) In AWS Amplify → Hosting → New app → Connect GitHub → select repo/branch.
3) Set environment variable: `NEXT_PUBLIC_API_URL` = your backend URL (e.g., `http://EC2_PUBLIC_DNS`).
4) Accept default Next.js build settings and deploy.
5) Copy the Amplify URL to include in Deliverables.


