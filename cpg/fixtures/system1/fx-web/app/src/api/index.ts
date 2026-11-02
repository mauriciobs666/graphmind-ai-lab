import { request } from "./client";

export const api = {
  search: {
    query: () => request("GET", "/api/search"),
  },
};
