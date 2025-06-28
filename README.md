# Velero Manager

A Kubernetes-native application that provides a centralized, user-friendly interface for managing backup and restore operations of microservices using Velero.

## Overview

Velero Manager simplifies disaster recovery and state management for Kubernetes workloads by providing an intuitive, secure, and efficient interface for Velero operations. The application leverages OAuth proxy for authentication and authorization, ensuring users can only manage namespaces where they have admin privileges.

## Features

- **OAuth Proxy Integration**: Secure authentication using OAuth proxy headers
- **Namespace Filtering**: Automatic filtering based on Kubernetes RBAC (admin role required)
- **Backup Management**: Create and manage backups for all resources in a namespace
- **Restore Operations**: One-click restore from existing backups
- **Real-time Progress**: Live progress tracking for backup and restore operations
- **User-friendly Interface**: Clean, professional Streamlit-based UI

## Prerequisites

- Kubernetes cluster with RBAC enabled
- Velero v1.14.1 or compatible version installed in cluster
- OAuth proxy configured and operational
- Python 3.8+ for local development

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd velero-manager
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run src/main.py
   ```

### Kubernetes Deployment

1. **Configure Helm values**
   ```bash
   cp helm/velero-manager/values.yaml.example helm/velero-manager/values.yaml
   # Edit values.yaml with your configuration
   ```

2. **Deploy using Helm**
   ```bash
   helm install velero-manager helm/velero-manager/
   ```

## Architecture

- **Backend**: Python application
- **Frontend**: Streamlit framework
- **Velero Integration**: Direct binary execution
- **Kubernetes Integration**: API client for namespace and resource queries
- **Deployment**: Helm chart for Kubernetes deployment

## Security

- All operations performed with user's bearer token when possible, falling back to pod service account token
- No privilege escalation
- Audit logging of all backup/restore operations
- Secure handling of credentials and sensitive data

## Configuration

The application is configured through environment variables and OAuth proxy headers:

- `Authorization`: User's bearer token for API authentication
- `X-Forwarded-User`: User's login identifier
- `X-Forwarded-Preferred-Username`: User's display name
- `X-Forwarded-Groups`: User's group memberships

## Project Structure

```
velero-manager/
├── src/                    # Application source code
│   ├── main.py            # Streamlit application entry point
│   ├── auth/              # Authentication modules
│   ├── k8s/               # Kubernetes API clients
│   ├── velero/            # Velero integration
│   └── ui/                # UI components
├── tests/                 # Test files
├── docs/                  # Documentation
├── helm/                  # Helm chart for deployment
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

The project follows PEP 8 style guidelines. Use tools like `black` and `flake8` for code formatting and linting.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please use the GitHub issue tracker.