const path = require('path');
const Dotenv = require('dotenv-webpack');

const config = {
    mode: 'development',
    target: 'electron-preload',
    devtool: 'source-map',
    entry: './public/preload.ts',
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