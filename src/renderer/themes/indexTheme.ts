import buttonTheme from './components/buttonTheme';
import checkboxTheme from './components/checkboxTheme';
import inputTheme from './components/inputTheme';
import textareaTheme from './components/textareaTheme';
// Import selectTheme from "./components/selectTheme";

const toast = {
    style: {
      borderRadius: "md",
      background: "rgba(0,0,0,0.8)",
      color: "white",
    },
    position: "top",
    duration: 5000,
    isClosable: true,
  };

const indexTheme = {
    initialColorMode: 'dark',
    useSystemColorMode: false,
    components: {
        Button: buttonTheme,
        Checkbox: checkboxTheme,
        NumberInputField: inputTheme,
        Textarea: textareaTheme
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
        toast: {
          zIndex: 9999, // set the zIndex to a value higher than your portal
        },
      },
};
export default indexTheme;
