import argparse
import os
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('backend_name', help='Backend name (folder name)')
    parser.add_argument('-s', '--silent', action='store_true', help='Silent mode - no output')
    args = parser.parse_args()

    # Get files from GitHub API
    url = f"https://api.github.com/repos/Qiskit/qiskit/contents/qiskit/providers/fake_provider/backends/{args.backend_name}?ref=stable%2F0.46"
    response = requests.get(url)
    files = response.json()

    save_location = f"qiskit_backend_configs/{args.backend_name}"
    
    # Create output folder
    os.makedirs(save_location, exist_ok=True)
    
    # Download files containing keywords
    keywords = ['conf', 'defs', 'props']
    for file_info in files:
        filename = file_info['name']
        if any(keyword in filename.lower() for keyword in keywords):
            # Download file
            file_response = requests.get(file_info['download_url'])
            with open(save_location + f"/{filename}", 'wb') as f:
                f.write(file_response.content)
            if not args.silent:
                print(f"Downloaded: {filename}")

if __name__ == '__main__':
    main()
