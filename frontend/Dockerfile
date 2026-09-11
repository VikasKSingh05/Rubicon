FROM node:22.14.0-alpine3.21 AS build

WORKDIR /app

ARG VITE_API_URL=http://localhost:4000
ARG VITE_AGENT_URL=http://localhost:8001
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_AGENT_URL=$VITE_AGENT_URL

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:22.14.0-alpine3.21

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY server.js ./
COPY --from=build /app/dist ./dist

ENV NODE_ENV=production
EXPOSE 3000

CMD ["node", "server.js"]