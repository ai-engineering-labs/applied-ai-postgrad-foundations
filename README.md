# Módulo 2 — exemplos (um repositório, várias pastas)

Cada pasta é um **mini-projeto independente** com o próprio `package.json`. Instale e rode **de dentro da pasta** do exemplo.

| Pasta | Tema | Instalar | Executar |
|--------|------|----------|----------|
| [`exemplo-00/`](exemplo-00/) | TensorFlow.js no Node (`tfjs-node`), rede simples | `cd exemplo-00 && npm install` | `npm start` |
| [`exemplo-01-ecommerce-recomendations-template/`](exemplo-01-ecommerce-recomendations-template/) | E-commerce + recomendações com TensorFlow.js no browser | `cd exemplo-01-ecommerce-recomendations-template && npm install` | `npm start` (abre em **http://localhost:3001**) |

## Novos exemplos

Use o padrão `exemplo-NN-tema/` e um `"name"` único no `package.json` (por exemplo prefixo `modulo2-exemplo-NN-...`). Evite reutilizar a mesma porta em dois servidores locais.

## Raiz do repositório

Apenas este índice, [`.gitignore`](.gitignore) e as pastas de exemplo — sem `package.json` na raiz (opção “coleção de laboratórios”).
