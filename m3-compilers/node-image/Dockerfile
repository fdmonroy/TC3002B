# We use debian as our base distro for this container.
FROM debian:bullseye

# Point to archive repo for EOL debian bullseye packages
RUN echo "deb http://archive.debian.org/debian bullseye main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security bullseye-security main" >> /etc/apt/sources.list && \
    apt-get update -o Acquire::Check-Valid-Until=false

# Install some utilities needed by node, npm.
RUN apt-get install -y --allow-unauthenticated curl make g++

# Install Node.js and npm.
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

# Install required npm packages.
ADD package.json /package.json
RUN npm install

# Copy application source files
ADD source /src

# Set /src as the working directory for this container.
WORKDIR /src

# Open up external access to port 8082
EXPOSE 8082

# Run startup command.
CMD ["node", "/src/server.js"]
