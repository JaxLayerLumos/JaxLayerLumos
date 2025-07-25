# Contributing to JaxLayerLumos

First off, thanks for taking the time to contribute to JaxLayerLumos! ❤️

All types of contributions are encouraged and valued. This document provides guidelines for contributing to JaxLayerLumos. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project on GitHub!
> - Refer this project in your project's README or publications
> - Mention the project at local meetups and tell your friends/colleagues

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [I Have a Question](#i-have-a-question)
- [I Want to Contribute](#i-want-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Improving the Documentation](#improving-the-documentation)
- [Styleguides](#styleguides)
  - [Commit Messages](#commit-messages)
  - [Python Code Style](#python-code-style)
- [Development Setup](#development-setup)

## Code of Conduct

This project and everyone participating in it is governed by the [JaxLayerLumos Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [jungtaek.kim.mail@gmail.com].

## I Have a Question

Before you ask a question, please ensure you have:
- Read the available [Documentation for JaxLayerLumos](LINK_TO_YOUR_DOCUMENTATION_SITE_OR_README_SECTION).
- Searched for existing [GitHub Issues](https://github.com/JaxLayerLumos/JaxLayerLumos/issues) that might address your question.
- Searched the Internet for answers.

If you still need to ask a question, we recommend the following:
- Open an [Issue on GitHub](https://github.com/JaxLayerLumos/JaxLayerLumos/issues/new).
- Provide as much context as you can about what you're trying to achieve or what problem you're encountering.
- Provide project and platform versions (e.g., Python version, JAX version, operating system) if relevant.

We will try to address your question as soon as possible.

## I Want to Contribute

> ### Legal Notice
> When contributing to this project, you must agree that you have authored 100% of the content, that you have the necessary rights to the content, and that the content you contribute may be provided under the project's [MIT License](LICENSE).

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Please investigate carefully, collect information, and describe the issue in detail.

- **Ensure you are using the latest version** of JaxLayerLumos.
- **Verify the bug:** Determine if it's a bug in JaxLayerLumos or an issue with your setup or usage (e.g., incompatible environment components/versions). Please consult the [documentation](LINK_TO_YOUR_DOCUMENTATION_SITE_OR_README_SECTION).
- **Check existing issues:** Search [GitHub Issues](https://github.com/JaxLayerLumos/JaxLayerLumos/issues) to see if the bug has already been reported.
- **Collect information:**
    - Full stack trace (Traceback) if applicable.
    - Your operating system, Python version, JAX version, and JaxLayerLumos version.
    - Input parameters or a minimal code snippet that reproduces the issue.
    - The behavior you observed and the behavior you expected.
- **Can you reliably reproduce the issue?**

#### How Do I Submit a Good Bug Report?

We use GitHub Issues to track bugs.

- Open a new [Issue](https://github.com/JaxLayerLumos/JaxLayerLumos/issues/new).
- Use a clear and descriptive title.
- Explain the expected behavior and the actual behavior.
- Provide detailed reproduction steps. A minimal, reproducible example is highly appreciated.
- Include the information you collected (versions, stack trace, etc.).

Once filed:
- The project team will label the issue.
- We will attempt to reproduce the issue. If we cannot, we may ask for more information.
- If reproduced, the issue will be marked for a fix.

### Your First Code Contribution

New to JAX or electromagnetic simulations? Here are some tips:
- Feel free to ask questions on the issue tracker if you need clarification.
- Fork the repository and create a new branch for your changes.
- Ensure your code adheres to the [Styleguides](#styleguides).
- Add tests for any new functionality or bug fixes.
- Update documentation if you change existing behavior or add new features.
- Submit a Pull Request (PR) with a clear description of your changes.

### Improving the Documentation

Good documentation is crucial! If you find areas for improvement, typos, or missing information in the [documentation](LINK_TO_YOUR_DOCUMENTATION_SITE_OR_README_SECTION) or in code docstrings:
- You can open an issue to discuss the changes.
- Or, directly submit a Pull Request with your improvements.

## Styleguides

### Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally after the first line.
- Consider using [Conventional Commits](https://www.conventionalcommits.org/) for more structured messages, e.g., `feat: Add support for anisotropic materials`.

### Python Code Style

- Follow [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
- Use a formatter like Black and isort to maintain consistent code style. (Specify if you have a preferred setup).
- Write clear and concise docstrings for all modules, classes, and functions, following [PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/). We recommend NumPy style docstrings.

## Development Setup

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone [https://github.com/JaxLayerLumos/JaxLayerLumos.git](https://github.com/JaxLayerLumos/JaxLayerLumos.git)
    cd JaxLayerLumos
    ```
3.  **Set up a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
4.  **Install JaxLayerLumos in editable mode**:
    ```bash
    pip install -e .
    ```
5.  **Create a new branch** for your changes:
    ```bash
    git checkout -b name-of-your-feature-or-fix
    ```
6.  Make your changes, write tests, and ensure all tests pass.
7.  Commit your changes and push to your fork.
8.  Open a Pull Request against the `main` branch of the original JaxLayerLumos repository.

Thank you for contributing to JaxLayerLumos!
