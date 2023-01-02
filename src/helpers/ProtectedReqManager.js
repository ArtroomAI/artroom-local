import axios from 'axios';

export default class ProtectedReqManager {
    constructor() {
    }
    
    static access_token = '';
    static refresh_token = '';

    static set_access_token(input_access_token) {
        this.access_token = input_access_token;
    }

    static set_refresh_token(input_refresh_token) {
        this.refresh_token = input_refresh_token;
    }

   
    static make_request(endpoint, wasRefreshed=false) {
        //const [accessToken, setAccessToken] = useRecoilState(atom.accessToken);
        //const [refreshToken, setRefreshToken] = useRecoilState(atom.refreshToken);

        const header_info = {
            headers: {
                'Content-Type': 'application/json',
                'accept': 'application/json',
                'Authorization': 'Bearer ' + this.access_token
            }
        }


        return axios.get(
            endpoint,
            header_info
        ).then(response => {
            return response;
        }).catch(err => {
            console.log(err);
            if (err.response.data.detail === "invalid access token" && !wasRefreshed) {
                //use refresh token to generate new access/refresh tokens and retry request
                const ARTROOM_URL = process.env.REACT_APP_SERVER_URL;

                let data = { refresh_token: this.refresh_token }

                return axios.post(`${ARTROOM_URL}/generate_refresh_token`,data,{
                    headers: {
                        'Content-Type': 'application/json',
                        'accept': 'application/json'
                    }
                }).then(response => {
                    this.set_access_token(response.data.access_token);
                    this.set_refresh_token(response.data.refresh_token);

                    //recursively call make_request(), but now with updated access tokens
                    return this.make_request(endpoint, true);
                    
                }).catch(err => {
                    //throw new Error(err);
                    throw err;
                });
            } else {
                throw err;
            }
        });
    }
}