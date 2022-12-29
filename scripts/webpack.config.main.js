const path = require('path');

const config = {
    mode: 'development',
    target: 'electron-main',
    devtool: 'source-map',
    entry: './public/main.js',
    output: {
        globalObject: 'this',
        filename: 'main.js',
        path: path.resolve(__dirname, '../build'),
        publicPath: ''
    },
    module: {
        rules: [
            {
                test: /\.(m|j|t)s$/,
                exclude: /(node_modules|bower_components)/,
                use: {
                    loader: 'babel-loader'
                }
            }
        ]
    },
    resolve: {
        extensions: ['.ts', '.js']
    }
}

module.exports = config;