#!/usr/bin/env python3
"""
Quick setup script for pushing nano-vLLM Runpod Edition to GitHub/GitLab.
Run this script to get step-by-step instructions for your specific repository.
"""

import os
import sys
import subprocess
from pathlib import Path

def get_current_repo_info():
    """Get current repository information"""
    try:
        # Check if git is initialized
        result = subprocess.run(['git', 'status'], capture_output=True, text=True, cwd='.')
        if result.returncode != 0:
            return None, "Git repository not initialized"
        
        # Get current branch
        result = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, cwd='.')
        current_branch = result.stdout.strip()
        
        # Get remote repositories
        result = subprocess.run(['git', 'remote', '-v'], capture_output=True, text=True, cwd='.')
        remotes = result.stdout.strip()
        
        return current_branch, remotes
    except Exception as e:
        return None, f"Error: {e}"

def print_setup_instructions():
    """Print repository setup instructions"""
    print("=" * 60)
    print("🚀 nano-vLLM Runpod Edition - Repository Setup")
    print("=" * 60)
    
    current_branch, remotes = get_current_repo_info()
    
    if current_branch:
        print(f"📍 Current branch: {current_branch}")
    else:
        print("⚠️  Git repository not initialized")
    
    if remotes:
        print(f"🔗 Current remotes:\n{remotes}")
    else:
        print("🔗 No remotes configured")
    
    print("\n" + "=" * 60)
    print("📋 STEP-BY-STEP SETUP GUIDE")
    print("=" * 60)
    
    print("\n1️⃣  CREATE REMOTE REPOSITORY:")
    print("   • Go to GitHub: https://github.com/new")
    print("   • Repository name: nano-vllm-runpod")
    print("   • Description: Production-ready fork of nano-vLLM optimized for Runpod serverless deployment")
    print("   • Keep it Private initially (recommended)")
    print("   • Don't initialize with README (you already have one)")
    
    print("\n2️⃣  CONFIGURE REMOTE ORIGIN:")
    print("   cd /Users/rachfop/nano/nano-vllm-runpod")
    print("   git remote add origin https://github.com/YOUR_USERNAME/nano-vllm-runpod.git")
    
    print("\n3️⃣  PUSH TO REMOTE:")
    print("   git push -u origin main")
    
    print("\n4️⃣  UPDATE CONFIGURATION FILES:")
    print("   • Update .runpod/hub.json with your repository URL")
    print("   • Update pyproject.toml with your author information")
    print("   • Update .github/workflows/deploy.yml if needed")
    
    print("\n5️⃣  SET UP SECRETS (for GitHub Actions):")
    print("   • RUNPOD_API_KEY: Your Runpod API key")
    print("   • CONTAINER_REGISTRY_TOKEN: If using private registry")
    
    print("\n" + "=" * 60)
    print("🔧 OPTIONAL COMMANDS")
    print("=" * 60)
    
    print("\n📊 Check repository status:")
    print("   git status")
    
    print("\n📝 View commit history:")
    print("   git log --oneline -n 5")
    
    print("\n🧪 Test configuration:")
    print("   python test_config.py")
    
    print("\n🐳 Build Docker image:")
    print("   docker build -t nano-vllm-runpod .")
    
    print("\n📚 View setup documentation:")
    print("   cat REPOSITORY_SETUP.md")
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    
    print("\n✅ After pushing to remote:")
    print("   1. Enable GitHub Actions in repository settings")
    print("   2. Add repository secrets (RUNPOD_API_KEY)")
    print("   3. Test deployment with a small change")
    print("   4. Configure repository topics/tags")
    print("   5. Make repository public when ready")
    
    print("\n📱 Repository URLs to update:")
    print("   • GitHub: https://github.com/YOUR_USERNAME/nano-vllm-runpod")
    print("   • Documentation: https://github.com/YOUR_USERNAME/nano-vllm-runpod/blob/main/README.md")
    print("   • Issues: https://github.com/YOUR_USERNAME/nano-vllm-runpod/issues")
    
    print("\n" + "=" * 60)

def main():
    """Main function"""
    print_setup_instructions()
    
    print("\n💡 TIP: Copy the commands above and paste them in your terminal!")
    print("💡 TIP: Replace 'YOUR_USERNAME' with your actual GitHub username!")
    
    # Check if git is configured
    try:
        result = subprocess.run(['git', 'config', '--global', 'user.name'], capture_output=True, text=True)
        if result.returncode != 0:
            print("\n⚠️  WARNING: Git user.name not configured!")
            print("   Run: git config --global user.name 'Your Name'")
        
        result = subprocess.run(['git', 'config', '--global', 'user.email'], capture_output=True, text=True)
        if result.returncode != 0:
            print("\n⚠️  WARNING: Git user.email not configured!")
            print("   Run: git config --global user.email 'your.email@example.com'")
    except Exception:
        pass
    
    print("\n🚀 Ready to deploy your nano-vLLM Runpod Edition!")

if __name__ == "__main__":
    main()