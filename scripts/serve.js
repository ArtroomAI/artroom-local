const { spawn } = require('child_process')
const path = require('path')

const del = import('del')

const webpack = require('webpack')
const WebpackDevServer = require('webpack-dev-server')

const configMain = require('./webpack.config.main')
const configPreload = require('./webpack.config.preload')
const configRenderer = require('./webpack.config.renderer')

const compilerMain = webpack(configMain)
const compilerPreload = webpack(configPreload)
const compilerRenderer = webpack(configRenderer)
const buildPath = path.join(__dirname, '../build')

let electronStarted = false

;(async () => {
  /**
   * Delete build dir
   */
  const { deleteSync } = await del
  deleteSync([buildPath], { force: true })

  /**
   * Start renderer dev server
   */
  const renderSrvOpts = {
    hot: true,
    host: 'localhost',
    port: 3000
  }

  const server = new WebpackDevServer(renderSrvOpts, compilerRenderer)
  await server.start()
  console.log(`> Dev server is listening on port ${renderSrvOpts.port}`)

  /**
   * Start Electron
   */
  const startElectron = () => {
    console.log('> Running electron')
    const electronPath = path.join(
      process.cwd(),
      'node_modules',
      '.bin',
      process.platform === 'win32' ? 'electron.cmd' : 'electron'
    )
    const electron = spawn(electronPath, [path.join(buildPath, 'main.js')], {
      stdio: 'inherit'
    })

    electron.on('exit', function () {
      process.exit(0)
    })
  }

  /**
   * Start Electron
   */
  const startPreload = () => {
    console.log('> Compiling Preload')

    compilerPreload.run((err, stats) => {
      console.log('> Preload compiled (3/3)')
    })
    compilerPreload.hooks.afterDone.tap('on-preload-build', startElectron)
  }

  /**
   * Start main
   */
  const startMain = stats => {
    console.log('> Renderer compiled')

    if (!electronStarted) {
      electronStarted = true
      console.log('> Compiling Main')
      compilerMain.run((err, stats) => {
        console.log('> Main compiled')
      })
      compilerMain.hooks.afterDone.tap('on-main-build', startPreload)
    }
  }

  server.compiler.hooks.afterEmit.tap('on-renderer-start', startMain)
})()
