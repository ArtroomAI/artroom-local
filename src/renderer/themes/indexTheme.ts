import { DeepPartial, Theme } from '@chakra-ui/react';

const indexTheme: DeepPartial<Theme> = {
    components: {
        Button: {
            // The styles all button have in common
            baseStyle: {
                borderRadius: '20'
            },
            // Two variants: outline and solid
            variants: {
                outline: {
                    borderColor: '#4f8ff8',
                    color: 'white'
                },
                solid: {
                    bg: '#4f8ff8',
                    color: 'white'
                }
            },
            // The default size and variant values
            defaultProps: {
                size: 'md',
                variant: 'solid',
                colorScheme: 'blue'
            }
        },
        Checkbox: {
            // The styles all button have in common
            baseStyle: () => ({
                borderRadius: 'base', // <-- border radius is same for all variants and sizes
                control: {
                    _checked: {
                        bg: '#4f8ff8',
                        color: '#fff',
                        borderColor: '#4f8ff8'
                    }
                }
            }),
            defaultProps: {
                size: 'md',
                variant: 'solid'
            }
        },
        Select: {
            "baseStyle": {
                field: {
                    "> option, > optgroup": {
                        "bg": "#080B16"
                    }
                }
            }
        }
    },
    styles: {
        global: {
        // Styles for the `body`
            html: {
                bg: '#080B16',
                color: 'white'
            },
            body: {
                bg: '#080B16',
                color: 'white'
            },
            borders: {
                borderWidth: '1px',
                borderRadius: '1px',
                borderStyle: 'solid',
                borderColor: '#FFFFFF20'
            }
        }
    },
    config: {
        initialColorMode: 'dark',
        useSystemColorMode: false
    },
    zIndices: {
        tooltip: 1000
    }
};
export default indexTheme;
