FROM node:22.14.0-alpine3.21 AS build

WORKDIR /app

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