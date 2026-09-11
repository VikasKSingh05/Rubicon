FROM node:22.14.0-alpine3.21

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY src ./src

ENV NODE_ENV=production
EXPOSE 8001

CMD ["node", "src/index.js"]
