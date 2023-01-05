const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const Dotenv = require('dotenv-webpack');

const config = {
    mode: 'development',
    entry: './public/renderer.ts',
    target: 'electron-renderer',
    devtool: 'source-map',
    module: {
        rules: [
            {
                test: /\.css$/,
                use: [{ loader: 'style-loader' }, { loader: 'css-loader' }],
            },
            {
                test: /\.(png|jpe?g|gif|ico)$/i,
                use: [{ loader: 'file-loader' }]
            },
            {
                test: /\.(m|j|t)s(x?)$/,
                exclude: /node_modules/,
                use: 'babel-loader'
            }
        ]
    },
    output: {
        globalObject: 'this',
        path: path.resolve(__dirname, '../build'),
        filename: 'renderer.js'
    },
    plugins: [
        new Dotenv(),
        new HtmlWebpackPlugin({
            template: './public/index.html'
        })
    ],
    resolve: {
      extensions: ['.ts', '.tsx', '.js', '.jsx']
    }
}

module.exports = config;
