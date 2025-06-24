#!/usr/bin/env python3
"""
Script to download all Mistral AI repositories and their GitHub issues
using repomix and GitHub API
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

# Configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "mistral-repos"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CONFIG_FILE = BASE_DIR / "config" / "mistral-repos.json"

# GitHub API headers
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "MistralAI-Repo-Downloader",
}

if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"


def load_repo_config() -> List[Dict[str, Any]]:
    """Load repository configuration from JSON file"""
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    return config["repositories"]


def run_repomix(repo_url: str, output_dir: Path) -> bool:
    """Run repomix to download and process a repository"""
    try:
        print(f"  Running repomix for {repo_url}...")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set up environment with proper PATH
        env = os.environ.copy()
        # Ensure node/npm paths are in PATH
        node_paths = [
            "/Users/andrewtan/.nvm/versions/node/v22.16.0/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
        ]
        current_path = env.get("PATH", "")
        for node_path in node_paths:
            if node_path not in current_path:
                env["PATH"] = f"{node_path}:{current_path}"

        # Run repomix directly from node_modules (without npx)
        repomix_path = BASE_DIR / "node_modules" / ".bin" / "repomix"
        cmd = [
            str(repomix_path),
            "--remote",
            repo_url,
            "--output",
            str(output_dir / "repomix-output.xml"),
            "--style",
            "xml",
            "--include-diffs",
            "--no-security-check",  # Skip security check for faster processing
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=output_dir, env=env
        )

        if result.returncode == 0:
            print("  ✓ Repomix completed successfully")
            return True
        else:
            print(f"  ✗ Repomix failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"  ✗ Error running repomix: {e}")
        return False


def get_github_issues(repo_name: str, output_dir: Path) -> bool:
    """Fetch all GitHub issues for a repository"""
    try:
        print(f"  Fetching GitHub issues for {repo_name}...")

        # API endpoint for issues
        url = f"https://api.github.com/repos/mistralai/{repo_name}/issues"

        all_issues = []
        page = 1

        while True:
            params = {
                "state": "all",  # Get both open and closed issues
                "per_page": 100,
                "page": page,
            }

            response = requests.get(url, headers=HEADERS, params=params)

            if response.status_code == 200:
                issues = response.json()
                if not issues:  # No more issues
                    break

                all_issues.extend(issues)
                page += 1

                # Rate limiting
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 10:
                        print(f"  Rate limit low ({remaining}), waiting...")
                        time.sleep(60)

            elif response.status_code == 403:
                print("  Rate limited, waiting 60 seconds...")
                time.sleep(60)
                continue
            else:
                print(f"  ✗ Failed to fetch issues: {response.status_code}")
                break

        # Save issues to file
        issues_file = output_dir / "github-issues.json"
        with open(issues_file, "w") as f:
            json.dump(all_issues, f, indent=2)

        print(f"  ✓ Saved {len(all_issues)} issues")
        return True

    except Exception as e:
        print(f"  ✗ Error fetching issues: {e}")
        return False


def get_github_pulls(repo_name: str, output_dir: Path) -> bool:
    """Fetch all GitHub pull requests for a repository"""
    try:
        print(f"  Fetching GitHub pull requests for {repo_name}...")

        # API endpoint for pull requests
        url = f"https://api.github.com/repos/mistralai/{repo_name}/pulls"

        all_pulls = []
        page = 1

        while True:
            params = {
                "state": "all",  # Get both open and closed PRs
                "per_page": 100,
                "page": page,
            }

            response = requests.get(url, headers=HEADERS, params=params)

            if response.status_code == 200:
                pulls = response.json()
                if not pulls:  # No more PRs
                    break

                all_pulls.extend(pulls)
                page += 1

                # Rate limiting
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 10:
                        print(f"  Rate limit low ({remaining}), waiting...")
                        time.sleep(60)

            elif response.status_code == 403:
                print("  Rate limited, waiting 60 seconds...")
                time.sleep(60)
                continue
            else:
                print(f"  ✗ Failed to fetch pull requests: {response.status_code}")
                break

        # Save pull requests to file
        pulls_file = output_dir / "github-pulls.json"
        with open(pulls_file, "w") as f:
            json.dump(all_pulls, f, indent=2)

        print(f"  ✓ Saved {len(all_pulls)} pull requests")
        return True

    except Exception as e:
        print(f"  ✗ Error fetching pull requests: {e}")
        return False


def get_issue_comments(repo_name: str, issue_number: int) -> List[Dict[str, Any]]:
    """Fetch all comments for a specific issue"""
    try:
        url = f"https://api.github.com/repos/mistralai/{repo_name}/issues/{issue_number}/comments"

        all_comments = []
        page = 1

        while True:
            params = {"per_page": 100, "page": page}

            response = requests.get(url, headers=HEADERS, params=params)

            if response.status_code == 200:
                comments = response.json()
                if not comments:  # No more comments
                    break

                all_comments.extend(comments)
                page += 1

                # Rate limiting
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 5:
                        time.sleep(1)  # Short delay for comment requests

            elif response.status_code == 403:
                time.sleep(60)
                continue
            else:
                break

        return all_comments

    except Exception as e:
        print(f"    ✗ Error fetching comments for issue #{issue_number}: {e}")
        return []


def get_pull_comments(
    repo_name: str, pull_number: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch all comments for a specific pull request (both issue comments and review comments)"""
    try:
        # Get issue comments (general comments on the PR)
        issue_comments_url = f"https://api.github.com/repos/mistralai/{repo_name}/issues/{pull_number}/comments"
        issue_comments = []
        page = 1

        while True:
            params = {"per_page": 100, "page": page}
            response = requests.get(issue_comments_url, headers=HEADERS, params=params)

            if response.status_code == 200:
                comments = response.json()
                if not comments:
                    break
                issue_comments.extend(comments)
                page += 1

                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 5:
                        time.sleep(1)
            elif response.status_code == 403:
                time.sleep(60)
                continue
            else:
                break

        # Get review comments (code review comments)
        review_comments_url = f"https://api.github.com/repos/mistralai/{repo_name}/pulls/{pull_number}/comments"
        review_comments = []
        page = 1

        while True:
            params = {"per_page": 100, "page": page}
            response = requests.get(review_comments_url, headers=HEADERS, params=params)

            if response.status_code == 200:
                comments = response.json()
                if not comments:
                    break
                review_comments.extend(comments)
                page += 1

                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 5:
                        time.sleep(1)
            elif response.status_code == 403:
                time.sleep(60)
                continue
            else:
                break

        return {"issue_comments": issue_comments, "review_comments": review_comments}

    except Exception as e:
        print(f"    ✗ Error fetching comments for PR #{pull_number}: {e}")
        return {"issue_comments": [], "review_comments": []}


def get_all_issue_comments(repo_name: str, output_dir: Path) -> bool:
    """Fetch comments for all issues in a repository"""
    try:
        print(f"  Fetching issue comments for {repo_name}...")

        # Load existing issues
        issues_file = output_dir / "github-issues.json"
        if not issues_file.exists():
            print("  ✗ No issues file found, skipping comments")
            return False

        with open(issues_file, "r") as f:
            issues = json.load(f)

        # Filter out pull requests (they have 'pull_request' key)
        actual_issues = [issue for issue in issues if "pull_request" not in issue]

        print(f"    Found {len(actual_issues)} issues to fetch comments for...")

        all_issue_comments = {}
        for i, issue in enumerate(actual_issues):
            issue_number = issue["number"]
            print(
                f"    Fetching comments for issue #{issue_number} ({i+1}/{len(actual_issues)})"
            )

            comments = get_issue_comments(repo_name, issue_number)
            if comments:
                all_issue_comments[str(issue_number)] = comments

            # Small delay between requests
            time.sleep(0.5)

        # Save all issue comments
        comments_file = output_dir / "github-issue-comments.json"
        with open(comments_file, "w") as f:
            json.dump(all_issue_comments, f, indent=2)

        total_comments = sum(len(comments) for comments in all_issue_comments.values())
        print(
            f"  ✓ Saved {total_comments} comments from {len(all_issue_comments)} issues"
        )
        return True

    except Exception as e:
        print(f"  ✗ Error fetching issue comments: {e}")
        return False


def get_all_pull_comments(repo_name: str, output_dir: Path) -> bool:
    """Fetch comments for all pull requests in a repository"""
    try:
        print(f"  Fetching pull request comments for {repo_name}...")

        # Load existing pull requests
        pulls_file = output_dir / "github-pulls.json"
        if not pulls_file.exists():
            print("  ✗ No pulls file found, skipping comments")
            return False

        with open(pulls_file, "r") as f:
            pulls = json.load(f)

        print(f"    Found {len(pulls)} pull requests to fetch comments for...")

        all_pull_comments = {}
        for i, pull in enumerate(pulls):
            pull_number = pull["number"]
            print(f"    Fetching comments for PR #{pull_number} ({i+1}/{len(pulls)})")

            comments = get_pull_comments(repo_name, pull_number)
            if comments["issue_comments"] or comments["review_comments"]:
                all_pull_comments[str(pull_number)] = comments

            # Small delay between requests
            time.sleep(0.5)

        # Save all pull request comments
        comments_file = output_dir / "github-pull-comments.json"
        with open(comments_file, "w") as f:
            json.dump(all_pull_comments, f, indent=2)

        total_issue_comments = sum(
            len(pr["issue_comments"]) for pr in all_pull_comments.values()
        )
        total_review_comments = sum(
            len(pr["review_comments"]) for pr in all_pull_comments.values()
        )
        total_comments = total_issue_comments + total_review_comments

        print(
            f"  ✓ Saved {total_comments} comments ({total_issue_comments} issue + {total_review_comments} review) from {len(all_pull_comments)} PRs"
        )
        return True

    except Exception as e:
        print(f"  ✗ Error fetching pull request comments: {e}")
        return False


def create_repo_metadata(
    repo_config: Dict[str, Any],
    output_dir: Path,
    repomix_success: bool,
    issues_success: bool,
    pulls_success: bool,
    issue_comments_success: bool,
    pull_comments_success: bool,
):
    """Create metadata file for the repository"""
    metadata = {
        "repository": repo_config,
        "download_status": {
            "repomix_success": repomix_success,
            "issues_success": issues_success,
            "pulls_success": pulls_success,
            "issue_comments_success": issue_comments_success,
            "pull_comments_success": pull_comments_success,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "files": {
            "repomix_output": "repomix-output.xml" if repomix_success else None,
            "github_issues": "github-issues.json" if issues_success else None,
            "github_pulls": "github-pulls.json" if pulls_success else None,
            "github_issue_comments": (
                "github-issue-comments.json" if issue_comments_success else None
            ),
            "github_pull_comments": (
                "github-pull-comments.json" if pull_comments_success else None
            ),
        },
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def process_repository(repo_config: Dict[str, Any]) -> Dict[str, bool]:
    """Process a single repository"""
    repo_name = repo_config["name"]
    repo_url = repo_config["url"]

    print(f"\n📦 Processing repository: {repo_name}")
    print(f"   URL: {repo_url}")

    # Create output directory
    output_dir = DATA_DIR / repo_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download repository with repomix
    repomix_success = run_repomix(repo_url, output_dir)

    # Get GitHub issues
    issues_success = get_github_issues(repo_name, output_dir)

    # Get GitHub pull requests
    pulls_success = get_github_pulls(repo_name, output_dir)

    # Get issue comments
    issue_comments_success = get_all_issue_comments(repo_name, output_dir)

    # Get pull request comments
    pull_comments_success = get_all_pull_comments(repo_name, output_dir)

    # Create metadata
    create_repo_metadata(
        repo_config,
        output_dir,
        repomix_success,
        issues_success,
        pulls_success,
        issue_comments_success,
        pull_comments_success,
    )

    return {
        "repomix": repomix_success,
        "issues": issues_success,
        "pulls": pulls_success,
        "issue_comments": issue_comments_success,
        "pull_comments": pull_comments_success,
    }


def generate_summary_report(results: Dict[str, Dict[str, bool]]):
    """Generate a summary report of the download process"""
    total_repos = len(results)
    successful_repomix = sum(1 for r in results.values() if r["repomix"])
    successful_issues = sum(1 for r in results.values() if r["issues"])
    successful_pulls = sum(1 for r in results.values() if r["pulls"])
    successful_issue_comments = sum(
        1 for r in results.values() if r.get("issue_comments", False)
    )
    successful_pull_comments = sum(
        1 for r in results.values() if r.get("pull_comments", False)
    )

    summary = {
        "summary": {
            "total_repositories": total_repos,
            "successful_repomix_downloads": successful_repomix,
            "successful_issues_downloads": successful_issues,
            "successful_pulls_downloads": successful_pulls,
            "successful_issue_comments_downloads": successful_issue_comments,
            "successful_pull_comments_downloads": successful_pull_comments,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "detailed_results": results,
    }

    # Save summary
    summary_file = PROCESSED_DIR / "download-summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total repositories: {total_repos}")
    print(f"Successful repomix downloads: {successful_repomix}/{total_repos}")
    print(f"Successful issues downloads: {successful_issues}/{total_repos}")
    print(f"Successful pulls downloads: {successful_pulls}/{total_repos}")
    print(
        f"Successful issue comments downloads: {successful_issue_comments}/{total_repos}"
    )
    print(
        f"Successful pull comments downloads: {successful_pull_comments}/{total_repos}"
    )
    print(f"Summary saved to: {summary_file}")


def main():
    """Main function"""
    print("🚀 Starting Mistral AI Repository Download")
    print(f"Data directory: {DATA_DIR}")

    # Check if GitHub token is available
    if not GITHUB_TOKEN:
        print("⚠️  Warning: No GITHUB_TOKEN found. API rate limits will be lower.")
        print("   Set GITHUB_TOKEN environment variable for better performance.")

    # Load repository configuration
    repos = load_repo_config()
    print(f"📋 Found {len(repos)} repositories to process")

    # Create directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Process each repository
    results = {}
    for i, repo_config in enumerate(repos, 1):
        print(f"\n[{i}/{len(repos)}] Processing {repo_config['name']}...")
        results[repo_config["name"]] = process_repository(repo_config)

        # Small delay between repositories to be respectful
        time.sleep(2)

    # Generate summary report
    generate_summary_report(results)

    print("\n✅ Download process completed!")
    print(f"📁 Data saved in: {DATA_DIR}")
    print(f"📊 Summary saved in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
