# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to /app
COPY . /app

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Healthcheck to ensure the app is running
HEALTHCHECK --interval=600s --timeout=2s --retries=12 \
    CMD curl -f http://localhost:8501/ || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "ui.py", "--server.headless", "true", "--server.address", "0.0.0.0", "--browser.gatherUsageStats", "false"]

# Set the user to a non-root user (ensure user 1001 has necessary permissions)
USER 1001