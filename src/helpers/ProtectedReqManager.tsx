import axios from 'axios'
import { SetterOrUpdater } from 'recoil'
import { Dispatch, SetStateAction } from 'react'
import { CreateStandaloneToastReturn } from '@chakra-ui/react'

export default class ProtectedReqManager {
  constructor() {}

  static access_token = ''
  static refresh_token = ''
  static cloudMode: any
  static setCloudMode: SetterOrUpdater<boolean>
  static toast: CreateStandaloneToastReturn['toast']
  static loggedIn: any
  static setLoggedIn: Dispatch<SetStateAction<boolean>>

  static set_access_token(input_access_token: string) {
    this.access_token = input_access_token
  }

  static set_refresh_token(input_refresh_token: string) {
    this.refresh_token = input_refresh_token
  }

  static make_get_request(endpoint: string, wasRefreshed = false): any {
    //const [accessToken, setAccessToken] = useRecoilState(atom.accessToken);
    //const [refreshToken, setRefreshToken] = useRecoilState(atom.refreshToken);

    const header_info = {
      headers: {
        'Content-Type': 'application/json',
        accept: 'application/json',
        Authorization: 'Bearer ' + this.access_token
      }
    }

    return axios
      .get(endpoint, header_info)
      .then(response => {
        return response
      })
      .catch((err: any) => {
        console.log(err)
        if (
          err.response.data.detail === 'invalid access token' &&
          !wasRefreshed
        ) {
          //use refresh token to generate new access/refresh tokens and retry request
          const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL

          let data = { refresh_token: this.refresh_token }

          return axios
            .post(`${ARTROOM_URL}/generate_refresh_token`, data, {
              headers: {
                'Content-Type': 'application/json',
                accept: 'application/json'
              }
            })
            .then(response => {
              this.set_access_token(response.data.access_token)
              this.set_refresh_token(response.data.refresh_token)

              //recursively call make_request(), but now with updated access tokens
              return this.make_get_request(endpoint, true)
            })
            .catch(err => {
              //throw new Error(err);
              this.setLoggedIn(false)
              this.toast({
                title: 'Session Logged out due to inactivity',
                status: 'info',
                position: 'top',
                duration: 5000,
                isClosable: false
              })
              throw err
            })
        } else {
          throw err
        }
      })
  }

  static make_post_request(
    endpoint: string,
    body_data = {},
    wasRefreshed = false
  ): any {
    //const [accessToken, setAccessToken] = useRecoilState(atom.accessToken);
    //const [refreshToken, setRefreshToken] = useRecoilState(atom.refreshToken);

    // const header_info = {
    //     headers: {
    //         'Content-Type': 'application/json',
    //         'accept': 'application/json',
    //         'Authorization': 'Bearer ' + this.access_token
    //     }
    // }

    const header_info = {
      headers: {
        'Content-Type': 'application/json',
        Authorization: 'Bearer ' + this.access_token
      }
    }

    return axios
      .post(endpoint, body_data, header_info)
      .then(response => {
        return response
      })
      .catch((err: any) => {
        console.log(err)
        if (
          err.response.data.detail === 'invalid access token' &&
          !wasRefreshed
        ) {
          //use refresh token to generate new access/refresh tokens and retry request
          const ARTROOM_URL = process.env.REACT_APP_ARTROOM_URL

          let data = { refresh_token: this.refresh_token }

          return axios
            .post(`${ARTROOM_URL}/generate_refresh_token`, data, {
              headers: {
                'Content-Type': 'application/json',
                accept: 'application/json'
              }
            })
            .then(response => {
              this.set_access_token(response.data.access_token)
              this.set_refresh_token(response.data.refresh_token)

              //recursively call make_request(), but now with updated access tokens
              return this.make_post_request(endpoint, body_data, true)
            })
            .catch(err => {
              //throw new Error(err);
              this.setLoggedIn(false)
              this.toast({
                title: 'Session Logged out due to inactivity',
                status: 'info',
                position: 'top',
                duration: 5000,
                isClosable: false
              })
              throw err
            })
        } else {
          throw err
        }
      })
  }
}
