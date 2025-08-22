# Use a slim version of Python to keep the image small
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app
COPY colocaciones.py ./

# Expose the port that Streamlit runs on (corrected to match the entrypoint)
EXPOSE 8080

# Command to run the Streamlit app
ENTRYPOINT ["streamlit", "run", "colocaciones.py", "--server.port=8080", "--server.address=0.0.0.0"]