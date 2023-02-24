const path = require('path');
const Dotenv = require('dotenv-webpack');

const config = {
    mode: 'development',
    target: 'electron-main',
    devtool: 'source-map',
    entry: './src/main/main.ts',
    output: {
        globalObject: 'this',
        filename: 'main.js',
        path: path.resolve(__dirname, '../build'),
        publicPath: ''
    },
    plugins: [
        new Dotenv(),
    ],
    module: {
        rules: [
            {
                test: /\.(m|j|t)s(x?)$/,
                exclude: /node_modules/,
                use: 'babel-loader'
            }
        ]
    },
    resolve: {
        extensions: ['.ts', '.js']
    }
}

module.exports = config;