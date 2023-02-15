module.exports = {
  presets: [
    [
      '@babel/preset-typescript',
      {
        development: process.env.NODE_ENV === 'development'
      }
    ],
    [
      '@babel/preset-react',
      {
        development: process.env.NODE_ENV === 'development'
      }
    ]
  ]
}
