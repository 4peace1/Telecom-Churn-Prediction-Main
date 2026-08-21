# Security Policy

## Reporting a security issue

Do not open a public issue containing credentials, customer data, or other sensitive information. Contact the repository owner privately.

## Secrets

- Never commit API keys, tunnel tokens, passwords, or `.streamlit/secrets.toml`.
- Use environment variables or the deployment platform's secret manager.
- Revoke and rotate a secret immediately if it is exposed.

## Data use

The included dataset is a public educational sample. Do not add real customer information without appropriate authorisation, minimisation, access controls, and privacy review.
