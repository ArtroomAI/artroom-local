import buttonTheme from './components/buttonTheme';
import checkboxTheme from './components/checkboxTheme';
import inputTheme from './components/inputTheme';
import textareaTheme from './components/textareaTheme';
// Import selectTheme from "./components/selectTheme";

const indexTheme = {
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
        initialColorMode: 'dark',
        useSystemColorMode: false
    }
};
export default indexTheme;
