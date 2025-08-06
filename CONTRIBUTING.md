# Contributing to Video Summarization Web App

Thank you for your interest in contributing! Here's how you can help improve this project.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your forked repository
   ```bash
   git clone https://github.com/your-username/video-summarization-app.git
   cd video-summarization-app
   ```
3. **Set up** the development environment:
   ```bash
   # Create and activate virtual environment (Windows)
   python -m venv venv
   .\venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements-dev.txt
   
   # Install pre-commit hooks
   pre-commit install
   ```
4. **Run tests** to verify your setup:
   ```bash
   python -m pytest
   ```

## 🛠 Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. Make your changes following the code style guidelines

3. Write or update tests as needed

4. Run tests and fix any issues:
   ```bash
   python -m pytest
   ```

5. Commit your changes with a descriptive message:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # or
   git commit -m "fix: resolve issue with video processing"
   ```

6. Push your changes to your fork:
   ```bash
   git push origin your-branch-name
   ```

7. Open a **Pull Request** against the `main` branch

## 📝 Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Keep lines under 88 characters (Black's default line length)
- Use meaningful variable and function names

## 🧪 Testing

- Write unit tests for new features and bug fixes
- Ensure all tests pass before submitting a PR
- Use descriptive test function names that describe the behavior being tested
- Follow the Arrange-Act-Assert pattern in tests

## 📚 Documentation

- Update the README.md for significant changes
- Add docstrings to new functions and classes
- Document any new environment variables or configuration options

## 🐛 Reporting Issues

When reporting issues, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected vs. actual behavior
4. Environment details (OS, Python version, etc.)
5. Any relevant error messages or logs

## 🤝 Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## 📜 License

By contributing, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE) file.

## 🙏 Thank You!

Your contributions make open-source a fantastic place to learn, inspire, and create. Thank you for being part of our community!
