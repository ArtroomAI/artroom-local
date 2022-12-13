import {useState} from "react";
import {
    useDisclosure,
    Button,
    Modal,
    ModalOverlay,
    ModalContent,
    ModalCloseButton
} from "@chakra-ui/react";
import Login from './Login'
import SignUp from "./Signup";

const LoginPage = ({setLoggedIn}) => {
  const { isOpen, onOpen, onClose } = useDisclosure();
  
  const [signUp, setSignUp] = useState(false);

  return (
    <>
        <Button varient='outline' onClick={onOpen} aria-label='View'>Login </Button>
        <Modal isOpen={isOpen} onClose={onClose} motionPreset='slideInBottom' scrollBehavior='outside' size='4xl'>
        <ModalOverlay />
            <ModalContent bg='gray.900'>
                <ModalCloseButton />
                {signUp ? 
                <SignUp setSignUp={setSignUp}/>
                : 
                <Login setLoggedIn={setLoggedIn} setSignUp={setSignUp}/>
                }
            </ModalContent>
        </Modal>
    </>

  );
};

export default LoginPage;
