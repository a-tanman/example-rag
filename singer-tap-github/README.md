# Singer Tap GitHub Integration

This folder contains the configuration and setup information for extracting GitHub data using Singer's tap-github and target-duckdb.

## Overview

This setup extracts data from GitHub repositories using the Singer ecosystem, specifically:
- **tap-github**: Extracts data from GitHub API
- **target-duckdb**: Loads data into DuckDB database

## Python Dependencies

Install the required Python libraries:

```bash
pip install duckdb>=1.3.1 numpy>=2.3.1 tap-github>=3.1.0 target-duckdb
```

Or using uv:

```bash
uv add duckdb>=1.3.1 numpy>=2.3.1 tap-github>=3.1.0 target-duckdb
```

## Configuration Files

### 1. Singer Configuration (`singer_config.json`)

```json
{
  "access_token": "<GITHUB_TOKEN>",
  "repository": "mistralai/client-python mistralai/mistral-inference mistralai/mistral-finetune mistralai/mistral-common mistralai/platform-docs-public mistralai/cookbook mistralai/client-js mistralai/client-ts mistralai/mistral-evals mistralai/sagemaker-docs",
  "start_date": "2021-01-01T00:00:00Z",
  "request_timeout": 300,
  "base_url": "https://api.github.com"
}
```

### 2. Target Configuration (`target_config.json`)

```json
{
    "path": "db.duckdb",
    "default_target_schema": "main"
}
```

### 3. Properties File (`properties.json`)

See `properties.json`

## Usage Command

Run the complete pipeline with:

```bash
tap-github --config singer_config.json --properties properties.json | target-duckdb --config target_config.json
```

## Data Streams Available

The tap-github extracts the following data streams:

### Core Streams
- **commits**: Git commit data with SHA, messages, authors, committers
- **comments**: Issue and PR comments with body, reactions, user info
- **issues**: GitHub issues with labels, assignees, milestones, reactions
- **pull_requests**: Pull request data with head/base branches, reviewers
- **reviews**: PR review data with state, body, submitted_at
- **review_comments**: Line-by-line PR review comments

### Repository Management
- **assignees**: Repository assignees
- **releases**: Repository releases with assets
- **issue_labels**: Available labels for issues
- **issue_milestones**: Project milestones
- **collaborators**: Repository collaborators with permissions
- **stargazers**: Users who starred the repository

### Events and Activity
- **events**: Repository events (pushes, issues, PRs, etc.)
- **issue_events**: Specific issue-related events
- **commit_comments**: Comments on specific commits

### Project Management
- **projects**: Repository projects
- **project_columns**: Project board columns
- **project_cards**: Cards within project columns

### Team Management (for organizations)
- **teams**: Organization teams
- **team_members**: Members of teams
- **team_memberships**: Team membership details

### Pull Request Details
- **pr_commits**: Commits within pull requests

## Key Features

### Incremental Replication
Most streams support incremental replication using:
- `updated_at` field for most entities
- `created_at` field for events
- `submitted_at` field for reviews

### Repository Coverage
Currently configured to extract data from Mistral AI repositories:
- mistralai/client-python
- mistralai/mistral-inference
- mistralai/mistral-finetune
- mistralai/mistral-common
- mistralai/platform-docs-public
- mistralai/cookbook
- mistralai/client-js
- mistralai/client-ts
- mistralai/mistral-evals
- mistralai/sagemaker-docs

### Data Schema
Each stream includes:
- Automatic key properties for unique identification
- Repository context (`_sdc_repository` field)
- Rich metadata including user information, timestamps, and relationships
- Nested objects for complex data (reactions, permissions, etc.)

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install duckdb>=1.3.1 numpy>=2.3.1 tap-github>=3.1.0 target-duckdb
   ```

2. **Configure GitHub Access**:
   - Update `singer_config.json` with your GitHub personal access token
   - Modify the repository list as needed

3. **Run Extraction**:
   ```bash
   tap-github --config singer_config.json --properties properties_v2.json --state state.json | target-duckdb --config target_config.json
   ```

4. **Query Data**:
   The data will be available in `db.duckdb` with tables for each stream.
