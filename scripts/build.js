const path = require('path')

const webpack = require('webpack')
const del = import('del')

const configMain = require('./webpack.config.main')
const configPreload = require('./webpack.config.preload')
const configRenderer = require('./webpack.config.renderer')

const compiler = webpack([configMain, configPreload, configRenderer])

;(async () => {
  /**
   * Delete build and dist dirs
   */
  const { deleteSync } = await del
  deleteSync(
    [path.join(__dirname, '../build'), path.join(__dirname, '../dist')],
    { force: true }
  )

  /**
   * Build main
   */
  compiler.run((err, stats) => {
    console.log('> Building main')
  })
})()
