const buttonTheme = {
  // The styles all button have in common
  baseStyle: {
    borderRadius: '20',
  },
  // Two variants: outline and solid
  variants: {
    outline: {
      borderColor: '#4f8ff8',
      color: 'white',
    },
    solid: {
      bg: '#4f8ff8',
      color: 'white',
    },
  },
  // The default size and variant values
  defaultProps: {
    size: 'md',
    variant: 'solid',
    colorScheme: 'blue'
  },
}

export default buttonTheme;