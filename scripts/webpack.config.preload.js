const path = require('path');
const Dotenv = require('dotenv-webpack');

const config = {
    mode: process.env.NODE_ENV,
    target: 'electron-preload',
    devtool: 'source-map',
    entry: './src/preload/preload.ts',
    module: {
        rules: [
            {
                test: /\.(m|j|t)s(x?)$/,
                exclude: /node_modules/,
                use: 'babel-loader'
            }
        ]
    },
    plugins: [
        new Dotenv(),
    ],
    output: {
        path: path.resolve(__dirname, '../build'),
        filename: 'preload.js'
    },
    resolve: {
        extensions: ['.ts', '.js']
    }
}

module.exports = config;