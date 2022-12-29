const path = require('path');

const config = {
    mode: 'development',
    target: 'electron-preload',
    devtool: 'source-map',
    entry: './public/preload.js',
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
    output: {
        path: path.resolve(__dirname, '../build'),
        filename: 'preload.js'
    },
    resolve: {
        extensions: ['.ts', '.js']
    }
}

module.exports = config;