# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

---

## Reporting a Vulnerability

The SignBridge development team takes the security of participant biometric data, camera feeds, and cloud credentials seriously.

If you discover a security vulnerability (such as exposed credentials, unauthorized access endpoints, or data leakage):

1. **Do NOT open a public GitHub Issue.**
2. Send an email to the repository maintainers with:
   - Description of the vulnerability.
   - Steps to reproduce or proof-of-concept.
   - Potential impact.
3. We will acknowledge receipt within 48 hours and work with you to patch the issue before public disclosure.

---

## Sensitive Configuration & Environment

- Never commit your `.env` file to version control.
- Ensure all cloud storage credentials (Cloudinary API secrets, MongoDB Atlas connection strings) are stored as environment variables.

